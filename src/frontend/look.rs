//! Ojon's look: colours, type sizes and the few drawn parts every screen shares
//! (top bar, cards, buttons, pills). Screens draw with these; nothing here runs a job.

use ui_core::{Color, CornerRadius, Rect, Stroke, Vec2};
use ui_platform::PointerState;
use ui_render::{measure_text, CommandBuffer, DrawCommand};
use ui_themes::ThemePalette;
use ui_widgets::icon_cache::{IconAtlas, StudioIcon};

// Colour roles. One accent (teal) marks the thing to act on; violet belongs to the
// Uncensor flow only; green, amber and red are states.
pub const CANVAS: Color = Color::rgb(14, 18, 23);
pub const BAR: Color = Color::rgb(24, 31, 39);
pub const PANEL: Color = Color::rgb(27, 34, 43);
pub const RAISED: Color = Color::rgb(36, 45, 56);
pub const LINE: Color = Color::rgb(46, 57, 69);
pub const TEXT: Color = Color::rgb(232, 236, 241);
pub const MUTED: Color = Color::rgb(148, 160, 174);
pub const ACCENT: Color = Color::rgb(88, 191, 169);
pub const ACCENT_HOT: Color = Color::rgb(108, 207, 186);
pub const ACCENT_INK: Color = Color::rgb(10, 28, 25);
pub const VIOLET: Color = Color::rgb(176, 163, 232);
pub const GOOD: Color = Color::rgb(102, 201, 148);
pub const WARN: Color = Color::rgb(224, 177, 96);
pub const BAD: Color = Color::rgb(226, 112, 112);

pub const BAR_H: f32 = 56.0;
pub const RADIUS: f32 = 10.0;

/// The same colours for Kathamo's stock widgets, so older views match the new ones.
pub fn palette() -> ThemePalette {
    ThemePalette {
        canvas: CANVAS,
        surface: PANEL,
        elevated: RAISED,
        elevated_high: Color::rgb(44, 54, 66),
        border: LINE,
        text: TEXT,
        text_muted: MUTED,
        accent: ACCENT,
        accent_hover: ACCENT_HOT,
        danger: BAD,
        success: GOOD,
        warning: WARN,
    }
}

/// The app's places, in top-bar order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Screen {
    Home,
    Quantize,
    Uncensor,
    Queue,
    Tools,
    Setup,
}

impl Screen {
    pub const ALL: [Screen; 6] = [
        Screen::Home,
        Screen::Quantize,
        Screen::Uncensor,
        Screen::Queue,
        Screen::Tools,
        Screen::Setup,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Screen::Home => "Home",
            Screen::Quantize => "Quantize",
            Screen::Uncensor => "Uncensor",
            Screen::Queue => "Queue",
            Screen::Tools => "Tools",
            Screen::Setup => "Backends & Setup",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Button {
    /// The one thing to do next (teal fill).
    Primary,
    /// A real choice that is not the main one (outlined).
    Secondary,
    /// The Uncensor flow's action (violet outline).
    Violet,
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color::rgba(f(a.r, b.r), f(a.g, b.g), f(a.b, b.b), f(a.a, b.a))
}

/// Text cut with an ellipsis so it fits `width`.
pub fn fit(text: &str, size: f32, strong: bool, width: f32) -> String {
    if measure_text(text, size, strong) <= width {
        return text.to_string();
    }
    let mut out = String::new();
    for ch in text.chars() {
        out.push(ch);
        if measure_text(&out, size, strong) + measure_text("…", size, strong) > width {
            out.pop();
            break;
        }
    }
    out.push('…');
    out
}

/// Drawing and hit-testing for one frame.
pub struct Paint<'a> {
    pub cmd: &'a mut CommandBuffer,
    pub pointer: &'a PointerState,
    /// The visible area of a scrolled screen; nothing outside it can be hovered or clicked.
    pub clip: Option<Rect>,
}

impl<'a> Paint<'a> {
    pub fn new(cmd: &'a mut CommandBuffer, pointer: &'a PointerState) -> Self {
        Paint { cmd, pointer, clip: None }
    }
}

impl Paint<'_> {
    /// Start drawing a scrolled screen inside `area`.
    pub fn begin(&mut self, area: Rect) {
        self.cmd.push(DrawCommand::ClipPush { rect: area });
        self.clip = Some(area);
    }

    pub fn end(&mut self) {
        self.cmd.push(DrawCommand::ClipPop);
        self.clip = None;
    }

    pub fn hovered(&self, rect: Rect) -> bool {
        let at = self.pointer.position;
        rect.contains(at) && self.clip.is_none_or(|clip| clip.contains(at))
    }

    /// A small tick, for done steps and chosen cards.
    pub fn check(&mut self, icons: &mut IconAtlas, pos: Vec2, color: Color) {
        icons.render(self.cmd, StudioIcon::Check, Rect::from_min_size(pos, Vec2::new(16.0, 16.0)), color);
    }

    pub fn clicked(&self, rect: Rect) -> bool {
        self.pointer.primary_down && self.hovered(rect)
    }

    pub fn fill(&mut self, rect: Rect, fill: Color, radius: f32, border: Option<Color>) {
        self.cmd.push(DrawCommand::Rect {
            rect,
            fill,
            radius: CornerRadius::same(radius),
            stroke: border.map_or(Stroke::NONE, |c| Stroke::new(1.0, c)),
        });
    }

    pub fn line(&mut self, from: Vec2, to: Vec2, color: Color) {
        self.cmd.push(DrawCommand::Line {
            start: from,
            end: to,
            stroke: Stroke::new(1.0, color),
        });
    }

    pub fn text(&mut self, text: &str, pos: Vec2, size: f32, strong: bool, color: Color) {
        self.cmd.push(DrawCommand::Text {
            text: text.to_string(),
            pos,
            size,
            color,
            strong,
        });
    }

    /// Text cut to `width`; returns the drawn width.
    pub fn fitted(&mut self, text: &str, pos: Vec2, size: f32, strong: bool, width: f32, color: Color) -> f32 {
        let shown = fit(text, size, strong, width);
        let w = measure_text(&shown, size, strong);
        self.text(&shown, pos, size, strong, color);
        w
    }

    /// Text centred in `rect` both ways.
    pub fn centred(&mut self, text: &str, rect: Rect, size: f32, strong: bool, color: Color) {
        let w = measure_text(text, size, strong);
        let pos = Vec2::new(
            (rect.center().x - w * 0.5).round(),
            (rect.center().y - size * 0.40).round(),
        );
        self.text(text, pos, size, strong, color);
    }

    /// Text centred vertically in `rect`, starting at `x`.
    pub fn centred_y(&mut self, text: &str, x: f32, rect: Rect, size: f32, strong: bool, color: Color) {
        let pos = Vec2::new(x, (rect.center().y - size * 0.40).round());
        self.text(text, pos, size, strong, color);
    }

    /// A card: panel fill with a hairline; hover lifts the border a little.
    pub fn card(&mut self, rect: Rect) {
        let border = if self.hovered(rect) { mix(LINE, MUTED, 0.25) } else { LINE };
        self.fill(rect, PANEL, RADIUS + 2.0, Some(border));
    }

    pub fn button(&mut self, rect: Rect, label: &str, kind: Button) -> bool {
        let hot = self.hovered(rect);
        let (fill, border, ink) = match kind {
            Button::Primary => (if hot { ACCENT_HOT } else { ACCENT }, None, ACCENT_INK),
            Button::Secondary => (if hot { RAISED } else { PANEL }, Some(LINE), TEXT),
            Button::Violet => (
                if hot { mix(RAISED, VIOLET, 0.12) } else { RAISED },
                Some(mix(LINE, VIOLET, 0.45)),
                VIOLET,
            ),
        };
        self.fill(rect, fill, 8.0, border);
        self.centred(label, rect, 14.0, kind == Button::Primary, ink);
        self.clicked(rect)
    }

    /// A quiet text link in the accent colour; returns whether it was clicked.
    pub fn link(&mut self, label: &str, right: f32, y: f32) -> bool {
        let w = measure_text(label, 13.0, false);
        let rect = Rect::from_min_size(Vec2::new(right - w - 8.0, y - 6.0), Vec2::new(w + 16.0, 28.0));
        let color = if self.hovered(rect) { ACCENT_HOT } else { ACCENT };
        self.centred(label, rect, 13.0, false, color);
        self.clicked(rect)
    }

    /// A state word on a soft tint of its colour. Returns its width.
    pub fn pill(&mut self, label: &str, pos: Vec2, color: Color) -> f32 {
        let w = (measure_text(label, 12.0, false) + 20.0).round();
        let rect = Rect::from_min_size(pos, Vec2::new(w, 22.0));
        self.fill(rect, mix(PANEL, color, 0.16), 11.0, None);
        self.centred(label, rect, 12.0, false, color);
        w
    }

    /// An icon in a soft rounded square.
    pub fn icon_tile(&mut self, icons: &mut IconAtlas, rect: Rect, icon: StudioIcon, color: Color) {
        self.fill(rect, RAISED, RADIUS, None);
        let s = 22.0;
        let inner = Rect::from_min_size(
            Vec2::new(rect.center().x - s * 0.5, rect.center().y - s * 0.5),
            Vec2::new(s, s),
        );
        icons.render(self.cmd, icon, inner, color);
    }
}

pub fn pill_width(label: &str) -> f32 {
    (measure_text(label, 12.0, false) + 20.0).round()
}

/// Where each top-bar entry sits for a window `width` wide.
pub fn nav_rects(width: f32) -> Vec<(Screen, Rect)> {
    let widths: Vec<f32> = Screen::ALL
        .iter()
        .map(|s| (measure_text(s.label(), 14.0, false) + 26.0).round())
        .collect();
    let total: f32 = widths.iter().sum::<f32>() + 6.0 * (widths.len() as f32 - 1.0);
    // Centred, but never under the brand on the left or the actions on the right.
    let mut x = ((width - total) * 0.5).round().max(150.0);
    let mut out = Vec::new();
    for (screen, w) in Screen::ALL.iter().zip(widths) {
        out.push((*screen, Rect::from_min_size(Vec2::new(x, 12.0), Vec2::new(w, 32.0))));
        x += w + 6.0;
    }
    out
}

/// What the person asked for in the top bar this frame.
#[derive(Default)]
pub struct BarActions {
    pub go: Option<Screen>,
    pub process_model: bool,
    pub setup_assistant: bool,
}

/// The top bar: brand, the places, and "Process Model" (the app's one standing action).
pub fn top_bar(p: &mut Paint, width: f32, current: Screen, busy: bool) -> BarActions {
    let mut actions = BarActions::default();
    let bar = Rect::from_min_size(Vec2::ZERO, Vec2::new(width, BAR_H));
    p.fill(bar, BAR, 0.0, None);
    p.line(Vec2::new(0.0, BAR_H), Vec2::new(width, BAR_H), LINE);

    let mark = Rect::from_min_size(Vec2::new(24.0, 14.0), Vec2::new(28.0, 28.0));
    p.fill(mark, mix(BAR, ACCENT, 0.35), 7.0, Some(mix(BAR, ACCENT, 0.7)));
    p.centred("O", mark, 15.0, true, ACCENT_HOT);
    p.centred_y("Ojon", 62.0, bar, 18.0, true, TEXT);

    for (screen, rect) in nav_rects(width) {
        let on = screen == current;
        let hot = p.hovered(rect);
        if on {
            p.fill(rect, mix(BAR, ACCENT, 0.16), 8.0, None);
        } else if hot {
            p.fill(rect, RAISED, 8.0, None);
        }
        let color = if on { ACCENT_HOT } else if hot { TEXT } else { MUTED };
        p.centred(screen.label(), rect, 14.0, false, color);
        if screen == Screen::Queue && busy {
            let dot = Rect::from_min_size(Vec2::new(rect.max.x - 9.0, rect.min.y + 5.0), Vec2::new(6.0, 6.0));
            p.fill(dot, ACCENT, 3.0, None);
        }
        if p.clicked(rect) && !on {
            actions.go = Some(screen);
        }
    }

    let label = "+  Process Model";
    let w = (measure_text(label, 14.0, true) + 32.0).round();
    let rect = Rect::from_min_size(Vec2::new(width - 24.0 - w, 13.0), Vec2::new(w, 30.0));
    if p.button(rect, label, Button::Primary) {
        actions.process_model = true;
    }
    // Shown only when it cannot run into the places in the middle.
    let nav_end = nav_rects(width).last().map_or(0.0, |(_, r)| r.max.x);
    let assistant = "Setup Assistant";
    let aw = measure_text(assistant, 13.0, false) + 16.0;
    if rect.min.x - 16.0 - aw > nav_end + 16.0 {
        let link = Rect::from_min_size(Vec2::new(rect.min.x - 16.0 - aw, 13.0), Vec2::new(aw, 30.0));
        let color = if p.hovered(link) { TEXT } else { MUTED };
        p.centred(assistant, link, 13.0, false, color);
        if p.clicked(link) {
            actions.setup_assistant = true;
        }
    }
    actions
}

/// One past job, as Home shows it.
pub struct RecentJob {
    pub name: String,
    pub detail: String,
    pub state: String,
    pub source: String,
}

/// The running job, as Home shows it.
pub struct ActiveJob {
    pub name: String,
    pub stage: String,
    pub elapsed: String,
    pub steps_done: usize,
}

#[derive(Default)]
pub struct HomeActions {
    pub go: Option<Screen>,
    /// A recent model to load, and where to take it.
    pub open: Option<(String, Screen)>,
}

fn state_color(state: &str) -> Color {
    let s = state.to_ascii_lowercase();
    if s.contains("fail") || s.contains("error") || s.contains("cancel") {
        BAD
    } else if s.contains("run") || s.contains("progress") || s.contains("start") {
        ACCENT
    } else if s.contains("complete") || s.contains("done") || s.contains("ok") || s.contains("success") {
        GOOD
    } else {
        MUTED
    }
}

/// Home: the two things Ojon does, what is running, and what ran before.
pub fn home(
    p: &mut Paint,
    icons: &mut IconAtlas,
    area: Rect,
    active: Option<&ActiveJob>,
    recent: &[RecentJob],
) -> HomeActions {
    let mut actions = HomeActions::default();
    let column = (area.width() - 48.0).min(976.0);
    let left = (area.center().x - column * 0.5).round();
    let right = left + column;
    let mut y = area.min.y + 44.0;

    let title = Rect::from_min_size(Vec2::new(left, y), Vec2::new(column, 44.0));
    p.centred("Ojon", title, 36.0, true, TEXT);
    y += 52.0;
    let tag = Rect::from_min_size(Vec2::new(left, y), Vec2::new(column, 24.0));
    p.centred("Powerful AI. Less memory. More freedom.", tag, 17.0, false, MUTED);
    y += 56.0;

    // The two jobs Ojon does. Only Quantize is filled: it is the common first step.
    let gap = 24.0;
    let card_w = ((column - gap) * 0.5).floor();
    let card_h = 286.0;
    let cards = [
        (
            StudioIcon::Cpu,
            ACCENT,
            "Make a Model Smaller",
            ["Reduce model size while keeping the best practical", "quality, so large models run fast on your computer."],
            "Quantize",
            Button::Primary,
            Screen::Quantize,
        ),
        (
            StudioIcon::Sparkles,
            VIOLET,
            "Reduce Model Restrictions",
            ["Reduce unwanted refusals on harmless creative and", "technical requests. Your original file is kept."],
            "Uncensor",
            Button::Violet,
            Screen::Uncensor,
        ),
    ];
    for (i, (icon, tint, heading, body, action, kind, target)) in cards.into_iter().enumerate() {
        let rect = Rect::from_min_size(
            Vec2::new(left + i as f32 * (card_w + gap), y),
            Vec2::new(card_w, card_h),
        );
        p.card(rect);
        let x = rect.min.x + 32.0;
        let tile = Rect::from_min_size(Vec2::new(x, rect.min.y + 32.0), Vec2::new(48.0, 48.0));
        p.icon_tile(icons, tile, icon, tint);
        p.text(heading, Vec2::new(x, rect.min.y + 100.0), 21.0, true, TEXT);
        for (n, line) in body.iter().enumerate() {
            p.fitted(line, Vec2::new(x, rect.min.y + 136.0 + n as f32 * 23.0), 14.5, false, card_w - 64.0, MUTED);
        }
        let button = Rect::from_min_size(
            Vec2::new(x, rect.max.y - 32.0 - 44.0),
            Vec2::new(card_w - 64.0, 44.0),
        );
        if p.button(button, action, kind) {
            actions.go = Some(target);
        }
    }
    y += card_h + 48.0;

    // What is running now.
    let dot = Rect::from_min_size(Vec2::new(left, y + 3.0), Vec2::new(8.0, 8.0));
    p.fill(dot, if active.is_some() { ACCENT } else { LINE }, 4.0, None);
    p.text("Current Jobs", Vec2::new(left + 16.0, y), 17.0, true, TEXT);
    if p.link("View queue  ›", right, y) {
        actions.go = Some(Screen::Queue);
    }
    y += 36.0;
    let row = Rect::from_min_size(Vec2::new(left, y), Vec2::new(column, 72.0));
    p.card(row);
    match active {
        Some(job) => {
            let room = column - 300.0;
            p.fitted(&job.name, Vec2::new(row.min.x + 18.0, row.min.y + 15.0), 15.0, true, room, TEXT);
            p.fitted(&job.stage, Vec2::new(row.min.x + 18.0, row.min.y + 41.0), 13.0, false, room, MUTED);
            // Progress is shown as real steps finished and time spent, never a guessed percent.
            let steps = match job.steps_done {
                0 => "Starting".to_string(),
                1 => "1 step done".to_string(),
                n => format!("{n} steps done"),
            };
            let w = measure_text(&steps, 13.0, false);
            p.text(&steps, Vec2::new(row.max.x - 18.0 - w, row.min.y + 17.0), 13.0, false, ACCENT_HOT);
            let w = measure_text(&job.elapsed, 13.0, false);
            p.text(&job.elapsed, Vec2::new(row.max.x - 18.0 - w, row.min.y + 41.0), 13.0, false, MUTED);
            if p.clicked(row) {
                actions.go = Some(Screen::Queue);
            }
        }
        None => {
            p.centred_y("Nothing is running. Pick a model to start.", row.min.x + 18.0, row, 14.0, false, MUTED);
        }
    }
    y += 72.0 + 40.0;

    // What ran before.
    p.text("Recent Models", Vec2::new(left, y), 17.0, true, TEXT);
    y += 36.0;
    if recent.is_empty() {
        let empty = Rect::from_min_size(Vec2::new(left, y), Vec2::new(column, 64.0));
        p.card(empty);
        p.centred_y("Models you process will be listed here.", empty.min.x + 18.0, empty, 14.0, false, MUTED);
        return actions;
    }
    let gap = 16.0;
    let w = ((column - gap * 2.0) / 3.0).floor();
    for (i, job) in recent.iter().take(3).enumerate() {
        let rect = Rect::from_min_size(Vec2::new(left + i as f32 * (w + gap), y), Vec2::new(w, 132.0));
        if rect.min.y > area.max.y {
            break;
        }
        p.card(rect);
        let x = rect.min.x + 20.0;
        let pill_w = pill_width(&job.state);
        p.fitted(&job.name, Vec2::new(x, rect.min.y + 18.0), 15.0, true, w - 52.0 - pill_w, TEXT);
        p.pill(&job.state, Vec2::new(rect.max.x - 20.0 - pill_w, rect.min.y + 16.0), state_color(&job.state));
        p.fitted(&job.detail, Vec2::new(x, rect.min.y + 46.0), 13.0, false, w - 40.0, MUTED);
        let half = ((w - 40.0 - 8.0) * 0.5).floor();
        let quantize = Rect::from_min_size(Vec2::new(x, rect.max.y - 16.0 - 32.0), Vec2::new(half, 32.0));
        let uncensor = Rect::from_min_size(Vec2::new(x + half + 8.0, quantize.min.y), Vec2::new(half, 32.0));
        if p.button(quantize, "Quantize", Button::Secondary) {
            actions.open = Some((job.source.clone(), Screen::Quantize));
        }
        if p.button(uncensor, "Uncensor", Button::Secondary) {
            actions.open = Some((job.source.clone(), Screen::Uncensor));
        }
    }
    actions
}
