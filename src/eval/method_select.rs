//! Choose a model-editing candidate from REAL measured holdout scores only.
//! No inference or made-up score estimates. Missing metrics fail closed.

use serde_json::{json, Value};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
fn num(v: &Value, key: &str) -> io::Result<f64> {
    let n = v
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("missing measured {key}")))?;
    if !n.is_finite() || !(0.0..=1.0).contains(&n) {
        return Err(invalid(format!("{key} must be a finite fraction 0..1")));
    }
    Ok(n)
}
#[derive(Debug, Clone)]
struct Trial {
    name: String,
    method: Option<String>,
    refusal_rate: f64,
    benign_overrefusal_rate: Option<f64>,
    coding: f64,
    reasoning: f64,
    tools: f64,
    vision: Option<f64>,
    artifact: String,
    bytes: u64,
}
fn trial(v: &Value, vision: bool) -> io::Result<Trial> {
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .filter(|x| !x.is_empty() && x.len() < 128)
        .ok_or_else(|| invalid("candidate name missing"))?
        .to_owned();
    let method = v.get("method").and_then(Value::as_str).map(str::to_owned);
    let artifact = v
        .get("artifact")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("candidate artifact path missing"))?
        .to_owned();
    let bytes = v
        .get("bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("candidate measured bytes missing"))?;
    let benign_overrefusal_rate = if v
        .get("benign_overrefusal_rate")
        .is_some_and(|val| !val.is_null())
    {
        Some(num(v, "benign_overrefusal_rate")?)
    } else if v.get("overrefusal_rate").is_some_and(|val| !val.is_null()) {
        Some(num(v, "overrefusal_rate")?)
    } else {
        None
    };
    Ok(Trial {
        name,
        method,
        artifact,
        bytes,
        refusal_rate: num(v, "refusal_rate")?,
        benign_overrefusal_rate,
        coding: num(v, "coding")?,
        reasoning: num(v, "reasoning")?,
        tools: num(v, "tools")?,
        vision: if vision {
            Some(num(v, "vision")?)
        } else {
            None
        },
    })
}
#[derive(Debug)]
pub struct MethodChoice {
    pub report: Value,
}
impl MethodChoice {
    pub fn as_json(&self) -> &Value {
        &self.report
    }
}

/// Example input: {"schema":"ojon.edit_method_scores.v1","baseline":{...},
/// "max_capability_loss":0.02,"max_refusal_rate":0.05,
/// "candidates":[{"name":"magnitude",...},
/// {"name":"ara",...}]}.
/// "baseline" must contain the same measured metrics as candidates.
/// Select only candidates meeting individual baseline capability tolerances.
/// If max_refusal_rate is configured, refusal rate must also stay at/below it.
/// The threshold is opt-in for backwards compatibility; an unset gate is not
/// evidence that a model meets any uncensoring target.
/// Among qualifying candidates: lowest refusal rate, then smaller bytes.
/// If none passes, NO winner is selected.
pub fn choose_edit_method(input: &Value) -> io::Result<MethodChoice> {
    if input.get("schema").and_then(Value::as_str) != Some("ojon.edit_method_scores.v1") {
        return Err(invalid("expected ojon.edit_method_scores.v1"));
    }
    let loss = num(input, "max_capability_loss")?;
    let max_refusal_rate = if input.get("max_refusal_rate").is_some_and(|v| !v.is_null()) {
        Some(num(input, "max_refusal_rate")?)
    } else {
        None
    };

    let baseline = input
        .get("baseline")
        .ok_or_else(|| invalid("missing baseline measurements"))?;
    let vision = baseline.get("vision").is_some();
    let base = trial(baseline, vision)?;
    let arr = input
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("candidate array missing"))?;
    if arr.is_empty() || arr.len() > 100 {
        return Err(invalid("candidate count must be 1..100"));
    }
    let mut seen = std::collections::HashSet::new();
    let mut eligible = Vec::new();
    let mut ranked = Vec::new();
    for v in arr {
        let x = trial(v, vision)?;
        if !seen.insert(x.name.clone()) {
            return Err(invalid("duplicate method name"));
        }
        let mut reasons = Vec::new();
        if base.coding <= 0.0 || base.reasoning <= 0.0 || base.tools <= 0.0 {
            reasons.push(
                "baseline needs nonzero coding, reasoning and tool scores to verify retention"
                    .to_owned(),
            );
        }
        if x.refusal_rate > base.refusal_rate + 1e-12 {
            reasons.push("harmful refusal rate increased versus baseline".to_owned());
        }
        let benign_improved = matches!(
            (x.benign_overrefusal_rate, base.benign_overrefusal_rate),
            (Some(candidate), Some(original)) if candidate + 1e-12 < original
        );
        if x.refusal_rate + 1e-12 >= base.refusal_rate && !benign_improved {
            reasons.push("no measured refusal reduction versus baseline".to_owned());
        }
        for (name, measured, original) in [
            ("coding", x.coding, base.coding),
            ("reasoning", x.reasoning, base.reasoning),
            ("tools", x.tools, base.tools),
        ] {
            if measured + loss + 1e-12 < original {
                reasons.push(format!("{name} loss exceeds tolerance"));
            }
        }
        if vision && x.vision.unwrap() + loss + 1e-12 < base.vision.unwrap() {
            reasons.push("vision loss exceeds tolerance".to_owned());
        }
        if let Some(maximum) = max_refusal_rate {
            if x.refusal_rate > maximum + 1e-12 {
                reasons.push(format!(
                    "refusal rate exceeds configured maximum {maximum:.4}"
                ));
            }
        }
        if let (Some(cand_benign), Some(base_benign)) =
            (x.benign_overrefusal_rate, base.benign_overrefusal_rate)
        {
            if cand_benign > (base_benign + 0.15).max(0.25) + 1e-12 {
                reasons.push(format!(
                    "benign over-refusal rate {:.4} exceeds tolerance vs baseline {:.4}",
                    cand_benign, base_benign
                ));
            }
        }
        if reasons.is_empty() {
            eligible.push(x.clone());
        }
        ranked.push(json!({
            "name":x.name,
            "method":x.method.as_deref().unwrap_or(x.name.as_str()),
            "artifact":x.artifact,
            "refusal_rate":x.refusal_rate,
            "benign_overrefusal_rate":x.benign_overrefusal_rate,
            "bytes":x.bytes,"coding":x.coding,"reasoning":x.reasoning,"tools":x.tools,
            "vision":x.vision,"quality_pass":reasons.is_empty(),"rejection_reasons":reasons
        }));
    }
    eligible.sort_by(|a, b| {
        a.refusal_rate
            .total_cmp(&b.refusal_rate)
            .then(a.bytes.cmp(&b.bytes))
            .then(a.name.cmp(&b.name))
    });
    let selected = eligible.first();
    let status = if selected.is_some() {
        "SELECTED_FROM_MEASURED_SCORES"
    } else {
        "NO_QUALITY_PASSING_CANDIDATE"
    };
    Ok(MethodChoice {
        report: json!({
            "schema":"ojon.edit_method_choice.v1",
            "status":status,"max_capability_loss":loss,
            "max_refusal_rate":max_refusal_rate,
            "refusal_gate_configured":max_refusal_rate.is_some(),
            "vision_required":vision,
            "winner":selected.map(|v|json!({
                "method":v.name,"artifact":v.artifact,
                "refusal_rate":v.refusal_rate,"bytes":v.bytes
            })),
            "tested":ranked,
            "note":"Only supplied measured evidence was compared; Ojon has not rerun inference."
        }),
    })
}
pub fn choose_edit_method_file(path: &Path) -> io::Result<MethodChoice> {
    let size = fs::metadata(path)?.len();
    if size == 0 || size > 2 * 1024 * 1024 {
        return Err(invalid("method score file exceeds 2 MiB"));
    }
    let buf = fs::read(path)?;
    let value: Value =
        serde_json::from_slice(&buf).map_err(|e| invalid(format!("invalid score JSON: {e}")))?;
    choose_edit_method(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn score(name: &str, refusals: f64, coding: f64, vision: f64, bytes: u64) -> Value {
        json!({"name":name,"artifact":format!("{name}.gguf"),"bytes":bytes,
        "refusal_rate":refusals,"coding":coding,"reasoning":0.9,"tools":0.9,"vision":vision})
    }
    #[test]
    fn rejects_severe_quality_loss_even_if_refusals_are_zero() {
        let data = json!({"schema":"ojon.edit_method_scores.v1",
          "max_capability_loss":0.04,
          "baseline":score("base",0.75,0.9,0.9,1000),
          "candidates":[score("broken",0.0,0.3,0.2,100),
                        score("preserved",0.03,0.89,0.88,500)]});
        let chosen = choose_edit_method(&data).unwrap();
        assert_eq!(chosen.as_json()["winner"]["method"], "preserved");
        assert_eq!(chosen.as_json()["tested"][0]["quality_pass"], false);
    }
    #[test]
    fn refuses_to_select_candidate_with_unknown_vision() {
        let mut no_vis = score("unknown", 0.01, 0.95, 0.95, 100);
        no_vis.as_object_mut().unwrap().remove("vision");
        let data = json!({"schema":"ojon.edit_method_scores.v1",
        "max_capability_loss":0.03,"baseline":score("base",0.8,0.9,0.9,1000),
        "candidates":[no_vis]});
        assert!(choose_edit_method(&data).is_err());
    }
    #[test]
    fn threshold_blocks_high_refusal_even_when_capabilities_pass() {
        let data = json!({"schema":"ojon.edit_method_scores.v1",
          "max_capability_loss":0.04,"max_refusal_rate":0.05,
          "baseline":score("base",0.75,0.9,0.9,1000),
          "candidates":[score("still_refuses",0.40,0.9,0.9,100),
                        score("accepted",0.04,0.89,0.89,200)]});
        let result = choose_edit_method(&data).unwrap();
        assert_eq!(result.as_json()["winner"]["method"], "accepted");
        assert_eq!(result.as_json()["refusal_gate_configured"], true);
        assert_eq!(result.as_json()["tested"][0]["quality_pass"], false);
        assert!(result.as_json()["tested"][0]["rejection_reasons"][0]
            .as_str()
            .unwrap()
            .contains("refusal rate exceeds"));
    }
    #[test]
    fn impossible_refusal_gate_fails_input_validation() {
        let data = json!({"schema":"ojon.edit_method_scores.v1",
          "max_capability_loss":0.01,"max_refusal_rate":1.5,
          "baseline":score("base",0.70,0.9,0.9,1000),
          "candidates":[score("candidate",0.2,0.9,0.9,500)]});
        assert!(choose_edit_method(&data).is_err());
    }
    #[test]
    fn no_qualifying_method_is_honest() {
        let data = json!({"schema":"ojon.edit_method_scores.v1","max_capability_loss":0.0,
         "baseline":score("base",0.5,0.9,0.9,1000),
         "candidates":[score("bad",0.01,0.8,0.8,100)]});
        assert_eq!(
            choose_edit_method(&data).unwrap().as_json()["status"],
            "NO_QUALITY_PASSING_CANDIDATE"
        );
    }
    #[test]
    fn uninformative_baseline_cannot_publish_an_edit() {
        let data = json!({"schema":"ojon.edit_method_scores.v1","max_capability_loss":0.02,
            "baseline":score("base",0.0,0.0,0.9,1000),
            "candidates":[score("edited",0.0,0.0,0.9,1000)]});
        let result = choose_edit_method(&data).unwrap();
        assert_eq!(result.as_json()["winner"], Value::Null);
        assert_eq!(result.as_json()["status"], "NO_QUALITY_PASSING_CANDIDATE");
    }
}
