//! Ojon's guided screens: Quantize, Uncensor, Queue, and the two overlays (pipeline and
//! setup assistant). Each takes a plain view of the app's state and returns what the
//! person asked for; the workstation owns the state and starts the real jobs.

use super::look::{
    self, accent, accent_hot, bad, canvas, fit, good, line, muted, panel, pill_width, raised, violet,
    warn, Button, Paint,
};
use ui_core::{Color, Rect, Vec2};
use ui_render::measure_text;
use ui_widgets::icon_cache::IconAtlas;

/// What is known about the chosen model.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    None,
    Checking,
    Ready,
    MetadataOnly,
}

pub struct ModelView {
    pub name: String,
    pub facts: String,
    pub state: ModelState,
}

/// The running job, if any, and the last status line.
pub struct JobView {
    pub busy: bool,
    pub stage: String,
    pub done: Vec<String>,
    pub elapsed: String,
    pub log: Vec<String>,
    pub status: String,
}

/// A finished run, already put into words.
pub struct ResultView {
    pub accepted: bool,
    pub headline: String,
    pub lines: Vec<String>,
    pub has_output: bool,
}

/// A plan waiting for the person's approval.
pub struct PlanView {
    pub ready: bool,
    pub headline: String,
    pub lines: Vec<String>,
}

fn column(area: Rect, max: f32) -> (f32, f32) {
    let width = (area.width() - 48.0).min(max);
    ((area.center().x - width * 0.5).round(), width)
}

/// Words wrapped to `width`, at most `max` lines (the last one cut with an ellipsis).
fn wrap(text: &str, size: f32, width: f32, max: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        if measure_text(&candidate, size, false) <= width || current.is_empty() {
            current = candidate;
        } else {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.len() > max {
        let rest = lines.split_off(max - 1).join(" ");
        lines.push(fit(&rest, size, false, width));
    }
    lines
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color::rgba(f(a.r, b.r), f(a.g, b.g), f(a.b, b.b), 255)
}

fn page_title(p: &mut Paint, left: f32, y: f32, title: &str, sub: &str, width: f32) -> f32 {
    p.text(title, Vec2::new(left, y), 27.0, true, look::text());
    p.fitted(sub, Vec2::new(left, y + 38.0), 15.0, false, width, muted());
    y + 86.0
}

/// The numbered steps under a page title. `current` is the step being worked on.
fn stepper(p: &mut Paint, left: f32, y: f32, width: f32, steps: &[&str], current: usize, tint: Color) -> f32 {
    let each = width / steps.len() as f32;
    for (i, label) in steps.iter().enumerate() {
        let x = left + i as f32 * each;
        let badge = Rect::from_min_size(Vec2::new(x, y), Vec2::new(22.0, 22.0));
        let done = i < current;
        let on = i == current;
        p.fill(badge, if done || on { mix(panel(), tint, 0.28) } else { raised() }, 11.0, None);
        p.centred(&(i + 1).to_string(), badge, 12.0, true, if done || on { tint } else { muted() });
        let color = if on { look::text() } else if done { tint } else { muted() };
        p.centred_y(label, x + 30.0, badge, 14.0, on, color);
    }
    p.line(Vec2::new(left, y + 34.0), Vec2::new(left + width, y + 34.0), line());
    y + 72.0
}

fn step_heading(p: &mut Paint, left: f32, y: f32, width: f32, text: &str, note: &str) -> f32 {
    p.text(text, Vec2::new(left, y), 18.0, true, look::text());
    if !note.is_empty() {
        let room = width - measure_text(text, 18.0, true) - 24.0;
        let shown = fit(note, 13.0, false, room.max(40.0));
        let w = measure_text(&shown, 13.0, false);
        p.text(&shown, Vec2::new(left + width - w, y + 4.0), 13.0, false, muted());
    }
    y + 38.0
}

#[derive(Default)]
struct ModelActions {
    file: bool,
    folder: bool,
}

/// Step 1 on both flows: the chosen model, or the two ways to choose one.
fn model_card(
    p: &mut Paint,
    icons: &mut IconAtlas,
    left: f32,
    y: f32,
    width: f32,
    model: &ModelView,
    tint: Color,
) -> (f32, ModelActions) {
    let mut actions = ModelActions::default();
    let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 86.0));
    p.card(rect);
    let x = rect.min.x + 22.0;
    if model.state == ModelState::None {
        p.centred_y("No model chosen yet. Pick a GGUF file or a model folder.", x, rect, 14.0, false, muted());
        let folder = Rect::from_min_size(Vec2::new(rect.max.x - 22.0 - 150.0, rect.min.y + 24.0), Vec2::new(150.0, 38.0));
        let file = Rect::from_min_size(Vec2::new(folder.min.x - 12.0 - 150.0, folder.min.y), Vec2::new(150.0, 38.0));
        actions.file = p.button(file, "Choose file", Button::Primary);
        actions.folder = p.button(folder, "Choose folder", Button::Secondary);
        return (rect.max.y + 44.0, actions);
    }
    let links = 250.0;
    let name_w = p.fitted(&model.name, Vec2::new(x, rect.min.y + 20.0), 17.0, true, width - links - 200.0, look::text());
    let (word, color) = match model.state {
        ModelState::Ready => ("Compatible", good()),
        ModelState::Checking => ("Checking…", muted()),
        ModelState::MetadataOnly => ("Details only: no converter for this format", warn()),
        ModelState::None => ("", muted()),
    };
    let wx = x + name_w + 12.0;
    if model.state == ModelState::Ready {
        p.check(icons, Vec2::new(wx, rect.min.y + 22.0), color);
        p.text(word, Vec2::new(wx + 20.0, rect.min.y + 23.0), 13.0, false, color);
    } else {
        p.fitted(word, Vec2::new(wx, rect.min.y + 23.0), 13.0, false, width - links - name_w - 60.0, color);
    }
    p.fitted(&model.facts, Vec2::new(x, rect.min.y + 50.0), 13.0, false, width - links - 44.0, muted());
    let change = "Change model";
    let cw = measure_text(change, 13.0, false);
    let change_rect = Rect::from_min_size(Vec2::new(rect.max.x - 22.0 - cw - 8.0, rect.min.y + 28.0), Vec2::new(cw + 16.0, 30.0));
    let hot = p.hovered(change_rect);
    p.centred(change, change_rect, 13.0, false, if hot { look::text() } else { tint });
    actions.file = p.clicked(change_rect);
    let folder = "Use a folder";
    let fw = measure_text(folder, 13.0, false);
    let folder_rect = Rect::from_min_size(Vec2::new(change_rect.min.x - fw - 28.0, rect.min.y + 28.0), Vec2::new(fw + 16.0, 30.0));
    let hot = p.hovered(folder_rect);
    p.centred(folder, folder_rect, 13.0, false, if hot { look::text() } else { muted() });
    actions.folder = p.clicked(folder_rect);
    (rect.max.y + 44.0, actions)
}

/// One selectable card. Returns whether it was clicked.
#[allow(clippy::too_many_arguments)]
fn choice_card(
    p: &mut Paint,
    icons: &mut IconAtlas,
    rect: Rect,
    title: &str,
    tag: &str,
    body: &str,
    foot: &str,
    selected: bool,
    tint: Color,
) -> bool {
    let hot = p.hovered(rect);
    let border = if selected { tint } else if hot { mix(line(), muted(), 0.3) } else { line() };
    let fill = if selected { mix(panel(), tint, 0.07) } else { panel() };
    p.fill(rect, fill, 12.0, Some(border));
    let x = rect.min.x + 20.0;
    let room = rect.width() - 40.0;
    let tw = p.fitted(title, Vec2::new(x, rect.min.y + 18.0), 15.5, true, room - 26.0, look::text());
    if !tag.is_empty() && tw + pill_width(tag) + 40.0 < room {
        p.pill(tag, Vec2::new(x + tw + 10.0, rect.min.y + 17.0), warn());
    }
    if selected {
        p.check(icons, Vec2::new(rect.max.x - 36.0, rect.min.y + 19.0), tint);
    }
    let mut y = rect.min.y + 48.0;
    for line in wrap(body, 13.5, room, 4) {
        p.text(&line, Vec2::new(x, y), 13.5, false, muted());
        y += 20.0;
    }
    if !foot.is_empty() {
        p.line(Vec2::new(x, rect.max.y - 44.0), Vec2::new(rect.max.x - 20.0, rect.max.y - 44.0), line());
        p.fitted(foot, Vec2::new(x, rect.max.y - 32.0), 12.5, false, room, if selected { tint } else { muted() });
    }
    p.clicked(rect) && !selected
}

/// The live job: stage, time, finished steps, the last log lines, and Cancel.
fn job_panel(p: &mut Paint, left: f32, y: f32, width: f32, job: &JobView, tint: Color) -> (f32, bool) {
    let lines = job.log.len().min(2) as f32;
    let height = 96.0 + lines * 20.0;
    let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, height));
    p.fill(rect, panel(), 12.0, Some(mix(line(), tint, 0.5)));
    let x = rect.min.x + 22.0;
    let pw = p.pill("Working", Vec2::new(x, rect.min.y + 18.0), tint);
    let stage = if job.stage.trim().is_empty() { "Starting" } else { job.stage.as_str() };
    p.fitted(stage, Vec2::new(x + pw + 12.0, rect.min.y + 20.0), 15.0, true, width - pw - 300.0, look::text());
    let cancel = Rect::from_min_size(Vec2::new(rect.max.x - 22.0 - 96.0, rect.min.y + 14.0), Vec2::new(96.0, 32.0));
    let ew = measure_text(&job.elapsed, 13.0, false);
    p.text(&job.elapsed, Vec2::new(cancel.min.x - 16.0 - ew, rect.min.y + 22.0), 13.0, false, muted());
    let stop = p.button(cancel, "Cancel", Button::Secondary);
    let done = if job.done.is_empty() {
        "No step has finished yet.".to_string()
    } else {
        format!("Finished: {}", job.done.join("  ›  "))
    };
    p.fitted(&done, Vec2::new(x, rect.min.y + 56.0), 13.0, false, width - 44.0, muted());
    let mut ly = rect.min.y + 82.0;
    for line in job.log.iter().take(2) {
        p.fitted(line, Vec2::new(x, ly), 12.0, false, width - 44.0, mix(muted(), canvas(), 0.25));
        ly += 20.0;
    }
    (rect.max.y + 20.0, stop)
}

/// A headline with a state colour over plain lines, plus up to four buttons.
/// Returns the new y and the index of the button clicked.
#[allow(clippy::too_many_arguments)]
fn report_card(
    p: &mut Paint,
    left: f32,
    y: f32,
    width: f32,
    headline: &str,
    color: Color,
    lines: &[String],
    buttons: &[(&str, Button)],
) -> (f32, Option<usize>) {
    let height = 58.0 + lines.len() as f32 * 22.0 + if buttons.is_empty() { 8.0 } else { 60.0 };
    let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, height));
    p.fill(rect, panel(), 12.0, Some(mix(line(), color, 0.45)));
    let x = rect.min.x + 22.0;
    let dot = Rect::from_min_size(Vec2::new(x, rect.min.y + 24.0), Vec2::new(8.0, 8.0));
    p.fill(dot, color, 4.0, None);
    p.fitted(headline, Vec2::new(x + 18.0, rect.min.y + 18.0), 15.5, true, width - 62.0, look::text());
    let mut ly = rect.min.y + 52.0;
    for line in lines {
        p.fitted(line, Vec2::new(x, ly), 13.5, false, width - 44.0, muted());
        ly += 22.0;
    }
    let mut hit = None;
    let mut bx = rect.max.x - 22.0;
    for (i, (label, kind)) in buttons.iter().enumerate().rev() {
        let w = (measure_text(label, 14.0, *kind == Button::Primary) + 36.0).round();
        bx -= w;
        let b = Rect::from_min_size(Vec2::new(bx, rect.max.y - 20.0 - 36.0), Vec2::new(w, 36.0));
        if p.button(b, label, *kind) {
            hit = Some(i);
        }
        bx -= 10.0;
    }
    (rect.max.y + 20.0, hit)
}

fn status_line(p: &mut Paint, left: f32, y: f32, width: f32, status: &str) -> f32 {
    if status.trim().is_empty() {
        return y;
    }
    p.fitted(status, Vec2::new(left, y), 13.0, false, width, muted());
    y + 30.0
}

// ---------------------------------------------------------------- Quantize

pub struct QuantizeView<'a> {
    pub model: &'a ModelView,
    /// The app's goal index: 0 balanced, 1 smallest, 2 fastest, 3 maximum quality.
    pub goal: usize,
    /// The app's device index: 0 auto, 1 CPU, 2..4 a named GPU backend.
    pub device: usize,
    pub this_computer: String,
    pub output: String,
    pub job: &'a JobView,
    pub plan: Option<PlanView>,
    pub result: Option<ResultView>,
}

#[derive(Default)]
pub struct QuantizeActions {
    pub choose_file: bool,
    pub choose_folder: bool,
    pub goal: Option<usize>,
    pub device: Option<usize>,
    pub open_tools: bool,
    pub choose_output: bool,
    pub start: bool,
    pub preview: bool,
    pub approve: bool,
    pub cancel: bool,
    pub pipeline: bool,
    pub open_folder: bool,
    pub test_model: bool,
    pub export: bool,
    pub height: f32,
}

pub fn quantize(p: &mut Paint, icons: &mut IconAtlas, area: Rect, scroll: f32, v: &QuantizeView) -> QuantizeActions {
    let mut a = QuantizeActions::default();
    p.begin(area);
    let (left, width) = column(area, 848.0);
    let mut y = area.min.y + 44.0 - scroll;
    let ready = v.model.state == ModelState::Ready;

    y = page_title(p, left, y, "Quantize Model", "Make the model use less memory while keeping as much of its quality as is practical.", width);
    y = stepper(p, left, y, width, &["Choose Model", "Choose Goal", "Review & Start"], if ready { 2 } else { 0 }, accent());

    y = step_heading(p, left, y, width, "Step 1 — Choose Model", "");
    let (ny, m) = model_card(p, icons, left, y, width, v.model, accent());
    y = ny;
    a.choose_file = m.file;
    a.choose_folder = m.folder;

    y = step_heading(p, left, y, width, "Step 2 — Choose Goal", "Ojon picks the method for each part of the model");
    // The three goals a person chooses between; the numbers are each preset's real settings.
    let goals = [
        (3usize, "Best Quality", "", "Keeps as much quality as is practical while still reducing size. For code, hard reasoning and long documents.", "Starts at Q6_K · tightest quality limit"),
        (0usize, "Balanced", "Recommended", "Good quality with a real memory saving. The usual choice for day-to-day conversation and coding.", "Starts at Q5_K · standard quality limit"),
        (1usize, "Smallest Practical", "", "Fits into the least memory while every candidate still has to pass the quality check. For laptops and small GPUs.", "Starts at Q4_K · widest quality limit"),
    ];
    let gap = 16.0;
    let w = ((width - gap * 2.0) / 3.0).floor();
    for (i, (index, title, tag, body, foot)) in goals.iter().enumerate() {
        let rect = Rect::from_min_size(Vec2::new(left + i as f32 * (w + gap), y), Vec2::new(w, 186.0));
        if choice_card(p, icons, rect, title, tag, body, foot, v.goal == *index, accent()) {
            a.goal = Some(*index);
        }
    }
    y += 186.0 + 16.0;

    let run = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 124.0));
    p.card(run);
    p.text("Where will you run this model?", Vec2::new(run.min.x + 22.0, run.min.y + 18.0), 14.0, true, look::text());
    let iw = ((width - 44.0 - 24.0) / 3.0).floor();
    let places = [
        (Some(0usize), "This Computer", v.this_computer.as_str(), v.device == 0),
        (Some(1usize), "CPU Only", "Works on any computer, slower", v.device == 1),
        (None, "A Specific GPU Backend", "CUDA, SYCL or Vulkan: set in Tools", v.device >= 2),
    ];
    for (i, (index, title, sub, on)) in places.iter().enumerate() {
        let rect = Rect::from_min_size(Vec2::new(run.min.x + 22.0 + i as f32 * (iw + 12.0), run.min.y + 48.0), Vec2::new(iw, 58.0));
        let hot = p.hovered(rect);
        let border = if *on { accent() } else if hot { mix(line(), muted(), 0.3) } else { line() };
        p.fill(rect, if *on { mix(panel(), accent(), 0.09) } else { panel() }, 8.0, Some(border));
        p.fitted(title, Vec2::new(rect.min.x + 14.0, rect.min.y + 10.0), 14.0, true, iw - 52.0, look::text());
        p.fitted(sub, Vec2::new(rect.min.x + 14.0, rect.min.y + 32.0), 12.5, false, iw - 28.0, muted());
        if *on {
            p.check(icons, Vec2::new(rect.max.x - 30.0, rect.min.y + 12.0), accent());
        }
        if p.clicked(rect) {
            match index {
                Some(i) if !*on => a.device = Some(*i),
                None => a.open_tools = true,
                _ => {}
            }
        }
    }
    y = run.max.y + 44.0;

    y = step_heading(p, left, y, width, "Step 3 — Review and Start", "");
    if v.job.busy {
        let (ny, stop) = job_panel(p, left, y, width, v.job, accent());
        y = ny;
        a.cancel = stop;
        if p.link("View pipeline  ›", left + width, y) {
            a.pipeline = true;
        }
        y += 34.0;
    } else {
        let review = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 176.0));
        p.card(review);
        let x = review.min.x + 22.0;
        let goal = goals.iter().find(|g| g.0 == v.goal).map_or("Fastest (set in Tools)", |g| g.1);
        let place = places.iter().find(|pl| pl.3).map_or("This Computer", |pl| pl.1);
        let rows = [("Model", v.model.name.as_str()), ("Goal", goal), ("Runs on", place), ("Saves to", v.output.as_str())];
        for (i, (label, value)) in rows.iter().enumerate() {
            let ry = review.min.y + 20.0 + i as f32 * 24.0;
            p.text(label, Vec2::new(x, ry), 13.5, false, muted());
            p.fitted(value, Vec2::new(x + 96.0, ry), 13.5, false, width - 44.0 - 96.0 - 120.0, look::text());
        }
        let out = "Change folder";
        let ow = measure_text(out, 13.0, false);
        let out_rect = Rect::from_min_size(Vec2::new(review.max.x - 22.0 - ow - 8.0, review.min.y + 86.0), Vec2::new(ow + 16.0, 26.0));
        let hot = p.hovered(out_rect);
        p.centred(out, out_rect, 13.0, false, if hot { look::text() } else { accent() });
        a.choose_output = p.clicked(out_rect);

        p.line(Vec2::new(x, review.max.y - 58.0), Vec2::new(review.max.x - 22.0, review.max.y - 58.0), line());
        let pipe = "View pipeline";
        let pw = measure_text(pipe, 13.0, false);
        let pipe_rect = Rect::from_min_size(Vec2::new(x - 8.0, review.max.y - 46.0), Vec2::new(pw + 16.0, 34.0));
        let hot = p.hovered(pipe_rect);
        p.centred(pipe, pipe_rect, 13.0, false, if hot { accent_hot() } else { accent() });
        a.pipeline = p.clicked(pipe_rect);
        let start = Rect::from_min_size(Vec2::new(review.max.x - 22.0 - 170.0, review.max.y - 47.0), Vec2::new(170.0, 36.0));
        let preview = Rect::from_min_size(Vec2::new(start.min.x - 10.0 - 130.0, start.min.y), Vec2::new(130.0, 36.0));
        if ready {
            a.start = p.button(start, "Start Quantizing", Button::Primary);
            a.preview = p.button(preview, "Preview plan", Button::Secondary);
        } else {
            p.fill(start, raised(), 8.0, None);
            p.centred("Start Quantizing", start, 14.0, false, muted());
            let hint = "Choose a model first";
            let hw = measure_text(hint, 13.0, false);
            p.text(hint, Vec2::new(start.min.x - 16.0 - hw, start.min.y + 10.0), 13.0, false, muted());
        }
        y = review.max.y + 20.0;
    }

    if let Some(plan) = &v.plan {
        let color = if plan.ready { accent() } else { warn() };
        let buttons: &[(&str, Button)] = if plan.ready && !v.job.busy { &[("Approve and run", Button::Primary)] } else { &[] };
        let (ny, hit) = report_card(p, left, y, width, &plan.headline, color, &plan.lines, buttons);
        y = ny;
        a.approve = hit.is_some();
    }
    if let Some(result) = &v.result {
        let color = if result.accepted { good() } else { warn() };
        let buttons: &[(&str, Button)] = if result.has_output {
            &[("Export report", Button::Secondary), ("Test model", Button::Secondary), ("Show in folder", Button::Primary)]
        } else {
            &[]
        };
        let (ny, hit) = report_card(p, left, y, width, &result.headline, color, &result.lines, buttons);
        y = ny;
        match hit {
            Some(0) => a.export = true,
            Some(1) => a.test_model = true,
            Some(_) => a.open_folder = true,
            None => {}
        }
    }
    y = status_line(p, left, y, width, &v.job.status);
    a.height = y + scroll - area.min.y + 40.0;
    p.end();
    a
}

// ---------------------------------------------------------------- Uncensor

pub struct UncensorView<'a> {
    pub model: &'a ModelView,
    /// 0 keep full precision, 1 Q4_K_M, 2 Q5_K_M, 3 Q6_K.
    pub after: usize,
    pub output: String,
    pub job: &'a JobView,
    pub plan: Option<PlanView>,
    pub result: Option<ResultView>,
}

#[derive(Default)]
pub struct UncensorActions {
    pub choose_file: bool,
    pub choose_folder: bool,
    pub after: Option<usize>,
    pub open_tools: bool,
    pub choose_output: bool,
    pub start: bool,
    pub preview: bool,
    pub approve: bool,
    pub cancel: bool,
    pub open_folder: bool,
    pub test_model: bool,
    pub export: bool,
    pub quantize_result: bool,
    pub height: f32,
}

pub fn uncensor(p: &mut Paint, icons: &mut IconAtlas, area: Rect, scroll: f32, v: &UncensorView) -> UncensorActions {
    let mut a = UncensorActions::default();
    p.begin(area);
    let (left, width) = column(area, 848.0);
    let mut y = area.min.y + 44.0 - scroll;
    let ready = v.model.state == ModelState::Ready;

    y = page_title(p, left, y, "Uncensor Model", "Reduce refusals on harmless requests, and measure that the model still does its work.", width);
    y = stepper(p, left, y, width, &["Select Model", "Check Method", "Process"], if ready { 2 } else { 0 }, violet());

    y = step_heading(p, left, y, width, "Step 1 — Select Model", "");
    let (ny, m) = model_card(p, icons, left, y, width, v.model, violet());
    y = ny;
    a.choose_file = m.file;
    a.choose_folder = m.folder;

    y = step_heading(p, left, y, width, "Step 2 — Check Method", "");
    let method = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 318.0));
    p.fill(method, panel(), 12.0, Some(mix(line(), violet(), 0.35)));
    let x = method.min.x + 24.0;
    let tw = p.pill("Recommended method", Vec2::new(x, method.min.y + 22.0), violet());
    p.text("Automatic refusal-direction search", Vec2::new(x + tw + 12.0, method.min.y + 22.0), 16.5, true, look::text());
    let mut ly = method.min.y + 58.0;
    for line in wrap(
        "Ojon finds the direction in the model that produces refusals, tries several ways of removing it, measures each one, and keeps a result only if it passes.",
        13.5,
        width - 48.0,
        3,
    ) {
        p.text(&line, Vec2::new(x, ly), 13.5, false, muted());
        ly += 20.0;
    }
    ly += 10.0;
    for promise in [
        "Your original model file is never changed.",
        "The edited model is written as a new file.",
        "Refusal rate and capability are measured before a result is accepted.",
    ] {
        p.check(icons, Vec2::new(x, ly), good());
        p.text(promise, Vec2::new(x + 24.0, ly), 13.5, false, look::text());
        ly += 24.0;
    }
    ly += 12.0;
    p.line(Vec2::new(x, ly), Vec2::new(method.max.x - 24.0, ly), line());
    ly += 14.0;
    p.text("After editing", Vec2::new(x, ly), 13.5, true, look::text());
    ly += 26.0;
    let options = ["Keep full precision", "Also make Q4_K_M", "Also make Q5_K_M", "Also make Q6_K"];
    let ow = ((width - 48.0 - 30.0) / 4.0).floor();
    for (i, label) in options.iter().enumerate() {
        let rect = Rect::from_min_size(Vec2::new(x + i as f32 * (ow + 10.0), ly), Vec2::new(ow, 36.0));
        let on = v.after == i;
        let hot = p.hovered(rect);
        let border = if on { violet() } else if hot { mix(line(), muted(), 0.3) } else { line() };
        p.fill(rect, if on { mix(panel(), violet(), 0.1) } else { panel() }, 8.0, Some(border));
        p.centred(&fit(label, 13.0, false, ow - 16.0), rect, 13.0, on, if on { look::text() } else { muted() });
        if p.clicked(rect) && !on {
            a.after = Some(i);
        }
    }
    let manual = "Manual trace edit is in Tools";
    let mw = measure_text(manual, 13.0, false);
    let manual_rect = Rect::from_min_size(Vec2::new(method.max.x - 24.0 - mw - 8.0, method.max.y - 40.0), Vec2::new(mw + 16.0, 28.0));
    let hot = p.hovered(manual_rect);
    p.centred(manual, manual_rect, 13.0, false, if hot { look::text() } else { muted() });
    a.open_tools = p.clicked(manual_rect);
    y = method.max.y + 44.0;

    y = step_heading(p, left, y, width, "Step 3 — Process", "");
    if v.job.busy {
        let (ny, stop) = job_panel(p, left, y, width, v.job, violet());
        y = ny;
        a.cancel = stop;
    } else {
        let review = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 132.0));
        p.card(review);
        let x = review.min.x + 22.0;
        let rows = [("Model", v.model.name.as_str()), ("Saves to", v.output.as_str())];
        for (i, (label, value)) in rows.iter().enumerate() {
            let ry = review.min.y + 20.0 + i as f32 * 24.0;
            p.text(label, Vec2::new(x, ry), 13.5, false, muted());
            p.fitted(value, Vec2::new(x + 96.0, ry), 13.5, false, width - 44.0 - 96.0 - 120.0, look::text());
        }
        let out = "Change folder";
        let ow = measure_text(out, 13.0, false);
        let out_rect = Rect::from_min_size(Vec2::new(review.max.x - 22.0 - ow - 8.0, review.min.y + 40.0), Vec2::new(ow + 16.0, 26.0));
        let hot = p.hovered(out_rect);
        p.centred(out, out_rect, 13.0, false, if hot { look::text() } else { violet() });
        a.choose_output = p.clicked(out_rect);
        let start = Rect::from_min_size(Vec2::new(review.max.x - 22.0 - 170.0, review.max.y - 54.0), Vec2::new(170.0, 36.0));
        let preview = Rect::from_min_size(Vec2::new(start.min.x - 10.0 - 130.0, start.min.y), Vec2::new(130.0, 36.0));
        if ready {
            a.start = p.button(start, "Start", Button::Primary);
            a.preview = p.button(preview, "Preview plan", Button::Secondary);
        } else {
            p.fill(start, raised(), 8.0, None);
            p.centred("Start", start, 14.0, false, muted());
            let hint = "Choose a model first";
            let hw = measure_text(hint, 13.0, false);
            p.text(hint, Vec2::new(start.min.x - 16.0 - hw, start.min.y + 10.0), 13.0, false, muted());
        }
        y = review.max.y + 20.0;
    }

    if let Some(plan) = &v.plan {
        let buttons: &[(&str, Button)] = if plan.ready && !v.job.busy { &[("Approve and run", Button::Primary)] } else { &[] };
        let (ny, hit) = report_card(p, left, y, width, &plan.headline, violet(), &plan.lines, buttons);
        y = ny;
        a.approve = hit.is_some();
    }
    if let Some(result) = &v.result {
        let color = if result.accepted { good() } else { warn() };
        let buttons: &[(&str, Button)] = if result.has_output {
            &[
                ("Export report", Button::Secondary),
                ("Test model", Button::Secondary),
                ("Quantize this result", Button::Secondary),
                ("Show in folder", Button::Primary),
            ]
        } else {
            &[]
        };
        let (ny, hit) = report_card(p, left, y, width, &result.headline, color, &result.lines, buttons);
        y = ny;
        match hit {
            Some(0) => a.export = true,
            Some(1) => a.test_model = true,
            Some(2) => a.quantize_result = true,
            Some(_) => a.open_folder = true,
            None => {}
        }
    }
    y = status_line(p, left, y, width, &v.job.status);
    a.height = y + scroll - area.min.y + 40.0;
    p.end();
    a
}

// ---------------------------------------------------------------- Queue

pub struct HistoryRow {
    pub name: String,
    pub detail: String,
    pub state: String,
    /// "Load result" or "Recover", when the job left something to open.
    pub action: Option<&'static str>,
}

#[derive(Default)]
pub struct QueueActions {
    pub cancel: bool,
    pub open: Option<usize>,
    pub pipeline: bool,
    pub height: f32,
}

fn state_words(state: &str) -> (String, Color) {
    let s = state.to_ascii_uppercase();
    let color = if s.contains("FAIL") || s.contains("ERROR") {
        bad()
    } else if s.contains("CANCEL") || s.contains("RECOVER") {
        warn()
    } else if s.contains("RUN") || s.contains("START") {
        accent()
    } else if s.contains("COMPLETE") || s.contains("DONE") || s.contains("SUCCESS") || s.contains("FOUND") || s.contains("SELECTED") {
        good()
    } else {
        muted()
    };
    // Status codes read as words: CHECKPOINT_RECOVERABLE -> Checkpoint recoverable.
    let lower = state.replace('_', " ").to_lowercase();
    let mut chars = lower.chars();
    let words = match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => "Unknown".to_string(),
    };
    (words, color)
}

pub fn queue(p: &mut Paint, area: Rect, scroll: f32, job: &JobView, name: &str, history: &[HistoryRow]) -> QueueActions {
    let mut a = QueueActions::default();
    p.begin(area);
    let (left, width) = column(area, 848.0);
    let mut y = area.min.y + 44.0 - scroll;
    y = page_title(p, left, y, "Queue", "The job that is running now, and every job this computer has run.", width);

    y = step_heading(p, left, y, width, "Running now", "");
    if job.busy {
        p.fitted(name, Vec2::new(left, y), 14.0, false, width, muted());
        y += 26.0;
        let (ny, stop) = job_panel(p, left, y, width, job, accent());
        y = ny;
        a.cancel = stop;
        if p.link("View pipeline  ›", left + width, y) {
            a.pipeline = true;
        }
        y += 40.0;
    } else {
        let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 64.0));
        p.card(rect);
        p.centred_y("Nothing is running.", rect.min.x + 22.0, rect, 14.0, false, muted());
        y = rect.max.y + 40.0;
    }

    let count = match history.len() {
        0 => String::new(),
        1 => "1 job".to_string(),
        n => format!("{n} jobs"),
    };
    y = step_heading(p, left, y, width, "History", &count);
    if history.is_empty() {
        let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 64.0));
        p.card(rect);
        p.centred_y("Finished jobs will be listed here.", rect.min.x + 22.0, rect, 14.0, false, muted());
        y = rect.max.y + 20.0;
    }
    for (i, row) in history.iter().enumerate() {
        let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 68.0));
        y = rect.max.y + 10.0;
        if rect.max.y < area.min.y || rect.min.y > area.max.y {
            continue;
        }
        p.card(rect);
        let x = rect.min.x + 22.0;
        let mut right = rect.max.x - 22.0;
        if let Some(action) = row.action {
            let w = (measure_text(action, 14.0, false) + 32.0).round();
            let b = Rect::from_min_size(Vec2::new(right - w, rect.min.y + 17.0), Vec2::new(w, 34.0));
            if p.button(b, action, Button::Secondary) {
                a.open = Some(i);
            }
            right = b.min.x - 14.0;
        }
        let (words, color) = state_words(&row.state);
        let pw = pill_width(&words);
        p.pill(&words, Vec2::new(right - pw, rect.min.y + 23.0), color);
        let room = right - pw - 16.0 - x;
        p.fitted(&row.name, Vec2::new(x, rect.min.y + 14.0), 15.0, true, room, look::text());
        p.fitted(&row.detail, Vec2::new(x, rect.min.y + 39.0), 13.0, false, room, muted());
    }
    a.height = y + scroll - area.min.y + 40.0;
    p.end();
    a
}

// ---------------------------------------------------------------- Models

/// One model file Ojon knows about.
pub struct ModelRow {
    pub name: String,
    pub folder: String,
    pub size: String,
    pub made_by_ojon: bool,
    pub missing: bool,
}

#[derive(Default)]
pub struct ModelsActions {
    /// Load this row's model and open the given flow.
    pub open: Option<(usize, look::Screen)>,
    pub folder: Option<usize>,
    pub rescan: bool,
    pub add: bool,
    pub height: f32,
}

pub fn models(p: &mut Paint, area: Rect, scroll: f32, rows: &[ModelRow]) -> ModelsActions {
    let mut a = ModelsActions::default();
    p.begin(area);
    let (left, width) = column(area, 848.0);
    let mut y = area.min.y + 44.0 - scroll;
    let add = Rect::from_min_size(Vec2::new(left + width - 124.0, y + 2.0), Vec2::new(124.0, 36.0));
    let again = Rect::from_min_size(Vec2::new(add.min.x - 10.0 - 110.0, add.min.y), Vec2::new(110.0, 36.0));
    a.add = p.button(add, "Add model", Button::Primary);
    a.rescan = p.button(again, "Look again", Button::Secondary);
    y = page_title(p, left, y, "Models", "Models you opened, models Ojon made, and GGUF files in your model folders.", width);
    if rows.is_empty() {
        let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 64.0));
        p.card(rect);
        p.centred_y("No models yet. Add one to start.", rect.min.x + 22.0, rect, 14.0, false, muted());
        y = rect.max.y + 20.0;
    }
    for (i, row) in rows.iter().enumerate() {
        let rect = Rect::from_min_size(Vec2::new(left, y), Vec2::new(width, 72.0));
        y = rect.max.y + 10.0;
        if rect.max.y < area.min.y || rect.min.y > area.max.y {
            continue;
        }
        p.card(rect);
        let x = rect.min.x + 22.0;
        let mut right = rect.max.x - 22.0;
        if !row.missing {
            for (label, target) in [("Uncensor", Some(look::Screen::Uncensor)), ("Quantize", Some(look::Screen::Quantize)), ("Folder", None)] {
                let w = (measure_text(label, 14.0, false) + 28.0).round();
                let b = Rect::from_min_size(Vec2::new(right - w, rect.min.y + 19.0), Vec2::new(w, 34.0));
                if p.button(b, label, Button::Secondary) {
                    match target {
                        Some(screen) => a.open = Some((i, screen)),
                        None => a.folder = Some(i),
                    }
                }
                right = b.min.x - 8.0;
            }
        }
        let tag = if row.missing {
            Some(("File not found", warn()))
        } else if row.made_by_ojon {
            Some(("Made by Ojon", accent()))
        } else {
            None
        };
        let room = right - 8.0 - x;
        let tag_w = tag.map_or(0.0, |(word, _)| pill_width(word) + 10.0);
        let name_w = p.fitted(&row.name, Vec2::new(x, rect.min.y + 15.0), 15.0, true, (room - tag_w).max(60.0), if row.missing { muted() } else { look::text() });
        if let Some((word, color)) = tag {
            p.pill(word, Vec2::new(x + name_w + 10.0, rect.min.y + 13.0), color);
        }
        let detail = if row.size.is_empty() { row.folder.clone() } else { format!("{}  ·  {}", row.size, row.folder) };
        p.fitted(&detail, Vec2::new(x, rect.min.y + 42.0), 13.0, false, room, muted());
    }
    a.height = y + scroll - area.min.y + 40.0;
    p.end();
    a
}
// ---------------------------------------------------------------- Overlays

fn overlay(p: &mut Paint, window: Rect, size: Vec2) -> Rect {
    p.fill(window, Color::rgba(6, 8, 11, 200), 0.0, None);
    let rect = Rect::from_min_size(
        Vec2::new((window.center().x - size.x * 0.5).round(), (window.center().y - size.y * 0.5).round().max(look::BAR_H + 8.0)),
        size,
    );
    p.fill(rect, look::bar(), 14.0, Some(line()));
    rect
}

fn close_cross(p: &mut Paint, rect: Rect) -> bool {
    let cross = Rect::from_min_size(Vec2::new(rect.max.x - 56.0, rect.min.y + 28.0), Vec2::new(32.0, 32.0));
    if p.hovered(cross) {
        p.fill(cross, raised(), 8.0, None);
    }
    p.centred("×", cross, 22.0, false, muted());
    p.clicked(cross)
}

/// One phase of a job, as the pipeline shows it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Waiting,
    Active,
    Done,
}

pub struct PipelineView<'a> {
    pub title: String,
    pub operation: String,
    pub phases: [(&'static str, &'static str, Phase); 4],
    pub job: &'a JobView,
}

/// The pipeline overlay; returns true when it should close.
pub fn pipeline(p: &mut Paint, icons: &mut IconAtlas, window: Rect, v: &PipelineView) -> bool {
    let rect = overlay(p, window, Vec2::new(896.0f32.min(window.width() - 48.0), 560.0));
    let x = rect.min.x + 26.0;
    let tw = p.pill("Pipeline", Vec2::new(x, rect.min.y + 30.0), accent());
    p.fitted(&v.title, Vec2::new(x + tw + 12.0, rect.min.y + 28.0), 19.0, true, rect.width() - tw - 130.0, look::text());
    p.fitted(&v.operation, Vec2::new(x, rect.min.y + 62.0), 13.5, false, rect.width() - 110.0, muted());
    let mut close = close_cross(p, rect);
    p.line(Vec2::new(rect.min.x, rect.min.y + 98.0), Vec2::new(rect.max.x, rect.min.y + 98.0), line());

    let band = Rect::from_min_max(Vec2::new(rect.min.x, rect.min.y + 99.0), Vec2::new(rect.max.x, rect.min.y + 300.0));
    p.fill(band, canvas(), 0.0, None);
    let each = (rect.width() - 52.0) / 4.0;
    let node_y = band.min.y + 52.0;
    p.line(Vec2::new(x + each * 0.5, node_y + 24.0), Vec2::new(x + each * 3.5, node_y + 24.0), line());
    let mut focus = 0;
    for (i, (name, what, phase)) in v.phases.iter().enumerate() {
        let cx = x + each * (i as f32 + 0.5);
        let node = Rect::from_min_size(Vec2::new(cx - 24.0, node_y), Vec2::new(48.0, 48.0));
        let (fill, border) = match phase {
            Phase::Done => (mix(panel(), good(), 0.18), mix(line(), good(), 0.6)),
            Phase::Active => (mix(panel(), accent(), 0.2), accent()),
            Phase::Waiting => (panel(), line()),
        };
        p.fill(node, fill, 12.0, Some(border));
        match phase {
            Phase::Done => p.check(icons, Vec2::new(cx - 8.0, node_y + 16.0), good()),
            Phase::Active => {
                focus = i;
                p.fill(Rect::from_min_size(Vec2::new(cx - 6.0, node_y + 18.0), Vec2::new(12.0, 12.0)), accent(), 6.0, None);
            }
            Phase::Waiting => p.fill(Rect::from_min_size(Vec2::new(cx - 5.0, node_y + 19.0), Vec2::new(10.0, 10.0)), line(), 5.0, None),
        }
        let nw = measure_text(name, 14.0, true);
        p.text(name, Vec2::new(cx - nw * 0.5, node_y + 62.0), 14.0, true, if *phase == Phase::Waiting { muted() } else { look::text() });
        let shown = fit(what, 12.0, false, each - 12.0);
        let ww = measure_text(&shown, 12.0, false);
        p.text(&shown, Vec2::new(cx - ww * 0.5, node_y + 84.0), 12.0, false, muted());
    }
    p.line(Vec2::new(rect.min.x, band.max.y), Vec2::new(rect.max.x, band.max.y), line());

    let mut y = band.max.y + 22.0;
    let (name, what, _) = v.phases[focus];
    p.text(if v.job.busy { "Now:" } else { "First:" }, Vec2::new(x, y), 14.0, true, look::text());
    p.text(name, Vec2::new(x + 52.0, y), 14.0, true, accent_hot());
    y += 28.0;
    p.fitted(what, Vec2::new(x, y), 13.5, false, rect.width() - 52.0, muted());
    y += 30.0;
    let detail = Rect::from_min_size(Vec2::new(x, y), Vec2::new(rect.width() - 52.0, 94.0));
    p.fill(detail, canvas(), 8.0, Some(line()));
    let mut lines: Vec<String> = Vec::new();
    if v.job.busy {
        lines.push(format!("Stage: {}", if v.job.stage.trim().is_empty() { "starting" } else { v.job.stage.as_str() }));
        lines.push(if v.job.done.is_empty() { "Finished: nothing yet".into() } else { format!("Finished: {}", v.job.done.join(", ")) });
        if let Some(line) = v.job.log.first() {
            lines.push(line.clone());
        }
    } else {
        lines.push("Not started.".into());
        lines.push(v.job.status.clone());
    }
    for (i, line) in lines.iter().take(3).enumerate() {
        p.fitted(line, Vec2::new(detail.min.x + 14.0, detail.min.y + 14.0 + i as f32 * 24.0), 12.5, i == 0, detail.width() - 28.0, if i == 0 { look::text() } else { muted() });
    }
    let done = Rect::from_min_size(Vec2::new(rect.max.x - 26.0 - 96.0, rect.max.y - 52.0), Vec2::new(96.0, 34.0));
    if p.button(done, "Close", Button::Secondary) {
        close = true;
    }
    close
}

pub struct SetupView {
    pub step: usize,
    pub system: String,
    pub processor: String,
    /// Name and memory line for each graphics card found.
    pub gpus: Vec<(String, String)>,
    /// The tools Ojon needs for its main jobs, and whether each was found.
    pub tools: Vec<(&'static str, &'static str, bool)>,
    pub output: String,
}

#[derive(Default)]
pub struct SetupActions {
    pub step: Option<usize>,
    pub close: bool,
    pub rescan: bool,
    pub get_tools: bool,
    pub choose_output: bool,
    pub open_setup_page: bool,
}

/// The setup assistant overlay: four short steps over real checks of this computer.
pub fn setup_assistant(p: &mut Paint, icons: &mut IconAtlas, window: Rect, v: &SetupView) -> SetupActions {
    let mut a = SetupActions::default();
    let rect = overlay(p, window, Vec2::new(768.0f32.min(window.width() - 48.0), 648.0));
    let x = rect.min.x + 26.0;
    let inner = rect.width() - 52.0;
    p.text("Let's prepare Ojon", Vec2::new(x, rect.min.y + 28.0), 21.0, true, look::text());
    p.text("A quick check of what Ojon needs to process your models.", Vec2::new(x, rect.min.y + 60.0), 14.5, false, muted());
    a.close = close_cross(p, rect);
    p.line(Vec2::new(rect.min.x, rect.min.y + 98.0), Vec2::new(rect.max.x, rect.min.y + 98.0), line());

    let steps = ["1. Your Computer", "2. Processing Tools", "3. Storage Location", "4. Ready to Begin"];
    let mut tx = x;
    for (i, label) in steps.iter().enumerate() {
        let w = (measure_text(label, 13.5, false) + 24.0).round();
        let tab = Rect::from_min_size(Vec2::new(tx, rect.min.y + 110.0), Vec2::new(w, 30.0));
        let on = i == v.step;
        if on {
            p.fill(tab, mix(look::bar(), accent(), 0.16), 8.0, None);
        }
        let hot = p.hovered(tab);
        p.centred(label, tab, 13.5, false, if on { accent_hot() } else if hot { look::text() } else { muted() });
        if p.clicked(tab) && !on {
            a.step = Some(i);
        }
        tx += w + 8.0;
    }
    p.line(Vec2::new(rect.min.x, rect.min.y + 152.0), Vec2::new(rect.max.x, rect.min.y + 152.0), line());
    let body = Rect::from_min_max(Vec2::new(rect.min.x, rect.min.y + 153.0), Vec2::new(rect.max.x, rect.max.y - 62.0));
    p.fill(body, canvas(), 0.0, None);
    let mut y = body.min.y + 26.0;

    let found = v.tools.iter().filter(|t| t.2).count();
    let tools_ready = v.tools.first().is_some_and(|t| t.2);
    let fact = |p: &mut Paint, rect: Rect, label: &str, value: &str, note: &str, note_color: Color| {
        p.fill(rect, panel(), 10.0, Some(line()));
        p.text(label, Vec2::new(rect.min.x + 16.0, rect.min.y + 14.0), 12.5, false, muted());
        p.fitted(value, Vec2::new(rect.min.x + 16.0, rect.min.y + 36.0), 15.0, true, rect.width() - 32.0, look::text());
        if !note.is_empty() {
            p.fitted(note, Vec2::new(rect.min.x + 16.0, rect.min.y + 60.0), 12.5, false, rect.width() - 32.0, note_color);
        }
    };
    match v.step {
        0 => {
            p.text("Detected Hardware", Vec2::new(x, y), 17.0, true, look::text());
            p.text("Ojon uses this to choose where a job runs and how big a model can be.", Vec2::new(x, y + 28.0), 13.5, false, muted());
            y += 62.0;
            let half = ((inner - 16.0) * 0.5).floor();
            fact(p, Rect::from_min_size(Vec2::new(x, y), Vec2::new(half, 78.0)), "Operating system", &v.system, "", muted());
            fact(p, Rect::from_min_size(Vec2::new(x + half + 16.0, y), Vec2::new(half, 78.0)), "Processor", &v.processor, "", muted());
            y += 94.0;
            if v.gpus.is_empty() {
                fact(p, Rect::from_min_size(Vec2::new(x, y), Vec2::new(inner, 90.0)), "Graphics", "No GPU reported by your tools", "Jobs run on the processor until a GPU build of the tools is set up.", warn());
            }
            for (i, (name, memory)) in v.gpus.iter().take(2).enumerate() {
                let w = if v.gpus.len() == 1 { inner } else { half };
                fact(p, Rect::from_min_size(Vec2::new(x + i as f32 * (half + 16.0), y), Vec2::new(w, 90.0)), "Graphics and memory", name, memory, accent());
            }
        }
        1 => {
            p.text("Processing Tools", Vec2::new(x, y), 17.0, true, look::text());
            p.text(&format!("{found} of {} found. Ojon drives these tools; it does not replace them.", v.tools.len()), Vec2::new(x, y + 28.0), 13.5, false, muted());
            y += 62.0;
            for (name, what, ok) in v.tools.iter().take(5) {
                let row = Rect::from_min_size(Vec2::new(x, y), Vec2::new(inner, 46.0));
                p.fill(row, panel(), 8.0, Some(line()));
                p.centred_y(name, row.min.x + 16.0, row, 14.0, true, look::text());
                p.centred_y(&fit(what, 13.0, false, inner - 360.0), row.min.x + 200.0, row, 13.0, false, muted());
                let (word, color) = if *ok { ("Found", good()) } else { ("Missing", warn()) };
                let pw = pill_width(word);
                p.pill(word, Vec2::new(row.max.x - 16.0 - pw, row.min.y + 12.0), color);
                y += 54.0;
            }
            let rescan = Rect::from_min_size(Vec2::new(x, body.max.y - 50.0), Vec2::new(110.0, 34.0));
            a.rescan = p.button(rescan, "Check again", Button::Secondary);
            let get = Rect::from_min_size(Vec2::new(rescan.max.x + 10.0, rescan.min.y), Vec2::new(190.0, 34.0));
            a.get_tools = p.button(get, "Get llama.cpp tools", Button::Secondary);
            let page = Rect::from_min_size(Vec2::new(get.max.x + 10.0, rescan.min.y), Vec2::new(150.0, 34.0));
            a.open_setup_page = p.button(page, "Set paths by hand", Button::Secondary);
        }
        2 => {
            p.text("Storage Location", Vec2::new(x, y), 17.0, true, look::text());
            p.text("New models and their reports are saved here. Originals are never changed.", Vec2::new(x, y + 28.0), 13.5, false, muted());
            y += 62.0;
            fact(p, Rect::from_min_size(Vec2::new(x, y), Vec2::new(inner, 78.0)), "Results folder", &v.output, "", muted());
            y += 94.0;
            let choose = Rect::from_min_size(Vec2::new(x, y), Vec2::new(150.0, 36.0));
            a.choose_output = p.button(choose, "Choose folder", Button::Secondary);
        }
        _ => {
            p.text("Ready to Begin", Vec2::new(x, y), 17.0, true, look::text());
            y += 40.0;
            let checks = [
                (!v.gpus.is_empty(), if v.gpus.is_empty() { "No GPU reported: jobs run on the processor.".to_string() } else { format!("Graphics: {}", v.gpus[0].0) }),
                (tools_ready, if tools_ready { "The quantizer was found.".to_string() } else { "The quantizer is missing: see Processing Tools.".to_string() }),
                (true, format!("Results go to: {}", v.output)),
            ];
            for (ok, line) in checks {
                if ok {
                    p.check(icons, Vec2::new(x, y), good());
                } else {
                    p.fill(Rect::from_min_size(Vec2::new(x + 4.0, y + 4.0), Vec2::new(8.0, 8.0)), warn(), 4.0, None);
                }
                p.fitted(&line, Vec2::new(x + 26.0, y), 14.0, false, inner - 26.0, look::text());
                y += 30.0;
            }
        }
    }
    if v.step == 0 {
        let note = Rect::from_min_size(Vec2::new(x, body.max.y - 62.0), Vec2::new(inner, 44.0));
        let ok = !v.gpus.is_empty();
        p.fill(note, mix(canvas(), if ok { good() } else { warn() }, 0.1), 8.0, Some(mix(line(), if ok { good() } else { warn() }, 0.4)));
        p.centred_y(
            if ok { "Your tools report a graphics card, so jobs can use it." } else { "No GPU is reported yet. Ojon still works on the processor, more slowly." },
            note.min.x + 16.0,
            note,
            13.5,
            false,
            look::text(),
        );
    }

    let next = Rect::from_min_size(Vec2::new(rect.max.x - 26.0 - 110.0, rect.max.y - 48.0), Vec2::new(110.0, 34.0));
    if v.step < 3 {
        if p.button(next, "Continue", Button::Primary) {
            a.step = Some(v.step + 1);
        }
    } else if p.button(next, "Done", Button::Primary) {
        a.close = true;
    }
    if v.step > 0 {
        let back = Rect::from_min_size(Vec2::new(x - 8.0, rect.max.y - 48.0), Vec2::new(64.0, 34.0));
        let hot = p.hovered(back);
        p.centred("Back", back, 14.0, false, if hot { look::text() } else { muted() });
        if p.clicked(back) {
            a.step = Some(v.step - 1);
        }
    }
    a
}
