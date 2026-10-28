//! Import local per-layer activation traces and build bounded Rust edit recipes.
//!
//! Trace collection is a separate model-runtime responsibility: this importer
//! NEVER claims to hook into an LLM/VLM itself. It accepts newline-delimited
//! JSON records from an instrumented inference backend and turns their means
//! into a validated ojon.abliteration.v1 recipe without buffering all samples.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};

const MAX_LINE: usize = 2 * 1024 * 1024;
const MAX_TENSORS: usize = 256;
const MAX_FEATURES: usize = 131072;
const MAX_SAMPLES: u64 = 65536;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
#[derive(Default)]
struct Pair {
    good_sum: Vec<f64>,
    bad_sum: Vec<f64>,
    good_count: u64,
    bad_count: u64,
}
#[derive(Debug)]
pub struct ActivationRecipe {
    pub tensors: usize,
    pub records: u64,
    pub recipe: Value,
}
impl ActivationRecipe {
    pub fn as_json(&self) -> Value {
        json!({"schema":"ojon.activation_receipt.v1",
            "tensor_count":self.tensors,"records":self.records,
            "recipe":self.recipe,"capture_engine":"EXTERNAL_INSTRUMENTED_RUNTIME",
            "model_inference":"NOT_RUN_BY_OJON",
            "refusal_quality":"NOT_EVALUATED"})
    }
}

/// Each JSONL line: {"tensor":"...o_proj.weight","class":"good"|"bad",
/// "values":[f32,...]}. Values must be recorded at the OUTPUT of the named
/// linear weight. Input and output features are not interchangeable.
pub fn compile_activation_recipe(
    path: &Path,
    strength: f64,
    preserve_row_norms: bool,
) -> io::Result<ActivationRecipe> {
    if !strength.is_finite() || strength <= 0.0 || strength > 1.0 {
        return Err(invalid("strength must be >0 and <=1"));
    }
    let reader = BufReader::new(File::open(path)?);
    let mut records = 0u64;
    let mut states = BTreeMap::<String, Pair>::new();
    for line in reader.split(b'\n') {
        let line = line?;
        if line.len() > MAX_LINE {
            return Err(invalid("activation JSONL line exceeds 2MiB"));
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let item: Value = serde_json::from_slice(&line)
            .map_err(|e| invalid(format!("invalid activation JSON: {e}")))?;
        let tensor = item
            .get("tensor")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("record missing tensor"))?;
        if tensor.len() > 512 || tensor.is_empty() {
            return Err(invalid("invalid tensor name"));
        }
        if !states.contains_key(tensor) && states.len() >= MAX_TENSORS {
            return Err(invalid("too many activation tensors"));
        }
        let class = item
            .get("class")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("record class must be good or bad"))?;
        let vals = item
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("record values must be numeric array"))?;
        if vals.is_empty() || vals.len() > MAX_FEATURES {
            return Err(invalid("invalid activation width"));
        }
        let state = states.entry(tensor.to_owned()).or_default();
        // Record opposite-class width before mutably borrowing the current
        // class; this also avoids aliasing across Rust match branches.
        let other_width = if class == "good" {
            state.bad_sum.len()
        } else {
            state.good_sum.len()
        };
        let (sum, count) = match class {
            "good" => (&mut state.good_sum, &mut state.good_count),
            "bad" => (&mut state.bad_sum, &mut state.bad_count),
            _ => return Err(invalid("class must be good or bad")),
        };
        if *count >= MAX_SAMPLES {
            return Err(invalid("per-class activation sample cap exceeded"));
        }
        if sum.is_empty() {
            sum.resize(vals.len(), 0.0);
        }
        if sum.len() != vals.len() {
            return Err(invalid("inconsistent feature width for class"));
        }
        // Cross-class feature dimensions must match, including if one class
        // was seen before the other.
        if other_width != 0 && other_width != vals.len() {
            return Err(invalid("good and bad activation dimensions differ"));
        }
        for (acc, v) in sum.iter_mut().zip(vals) {
            let x = v
                .as_f64()
                .ok_or_else(|| invalid("activation not numeric"))?;
            if !x.is_finite() {
                return Err(invalid("nonfinite activation"));
            }
            *acc += x;
            if !acc.is_finite() {
                return Err(invalid("activation accumulation overflow"));
            }
        }
        *count += 1;
        records += 1;
    }
    if records == 0 {
        return Err(invalid("activation trace empty"));
    }
    let mut edits = Vec::with_capacity(states.len());
    for (
        tensor,
        Pair {
            good_sum,
            bad_sum,
            good_count,
            bad_count,
        },
    ) in states
    {
        if good_count == 0 || bad_count == 0 {
            return Err(invalid(format!(
                "missing good/bad activation pair for {tensor}"
            )));
        }
        if good_sum.len() != bad_sum.len() {
            return Err(invalid("paired activation width mismatch"));
        }
        let direction: Vec<f64> = good_sum
            .iter()
            .zip(&bad_sum)
            .map(|(good, bad)| bad / bad_count as f64 - good / good_count as f64)
            .collect();
        let norm = direction.iter().fold(0.0_f64, |acc, &x| acc.hypot(x));
        if !norm.is_finite() || norm < 1e-30 {
            return Err(invalid(format!(
                "zero or invalid mean-difference direction for {tensor}"
            )));
        }
        edits.push(
            json!({"tensor":tensor,"direction":direction.iter().map(|x|x/norm).collect::<Vec<_>>(),
            "strength":strength}),
        );
    }
    let tensors = edits.len();
    Ok(ActivationRecipe {
        tensors,
        records,
        recipe: json!({
            "schema":"ojon.abliteration.v1",
            "preserve_row_norms":preserve_row_norms,
            "edits":edits
        }),
    })
}

/// Do not overwrite a recipe or write through symlink/unsafe output path.
/// The output JSON must be placed into a user-controlled existing directory.
pub fn save_activation_recipe(recipe: &ActivationRecipe, output: &Path) -> io::Result<PathBuf> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent.canonicalize()?;
    let name = output
        .file_name()
        .ok_or_else(|| invalid("output filename missing"))?;
    if output.extension().and_then(|s| s.to_str()) != Some("json") {
        return Err(invalid("recipe must be written as .json"));
    }
    let out = parent.join(name);
    let mut f = OpenOptions::new().write(true).create_new(true).open(&out)?;
    let bytes = serde_json::to_vec_pretty(&recipe.recipe)
        .map_err(|e| invalid(format!("recipe serialization: {e}")))?;
    if let Err(e) = f.write_all(&bytes).and_then(|_| f.sync_all()) {
        drop(f);
        let _ = fs::remove_file(&out);
        return Err(e);
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    fn setup(data: &str) -> (PathBuf, PathBuf) {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("ojon-activation-{}-{time}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let input = root.join("trace.jsonl");
        fs::write(&input, data).unwrap();
        (root, input)
    }
    #[test]
    fn streaming_pair_means_generates_weight_recipe() {
        let trace=concat!(
            "{\"tensor\":\"model.layers.0.self_attn.o_proj.weight\",\"class\":\"good\",\"values\":[1,2]}\n",
            "{\"tensor\":\"model.layers.0.self_attn.o_proj.weight\",\"class\":\"bad\",\"values\":[4,2]}\n",
            "{\"tensor\":\"model.layers.0.self_attn.o_proj.weight\",\"class\":\"good\",\"values\":[3,2]}\n",
            "{\"tensor\":\"model.layers.0.self_attn.o_proj.weight\",\"class\":\"bad\",\"values\":[6,2]}\n"
        );
        let (root, path) = setup(trace);
        let r = compile_activation_recipe(&path, 0.8, true).unwrap();
        assert_eq!(r.records, 4);
        assert_eq!(r.tensors, 1);
        assert_eq!(r.recipe["edits"][0]["direction"], json!([1.0, 0.0]));
        let out = root.join("recipe.json");
        save_activation_recipe(&r, &out).unwrap();
        assert!(out.is_file());
        assert!(save_activation_recipe(&r, &out).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rejects_malformed_unpaired_and_mismatched() {
        for value in [
            "{\"tensor\":\"w\",\"class\":\"good\",\"values\":[1]}\n",
            "{\"tensor\":\"w\",\"class\":\"good\",\"values\":[1]}\n{\"tensor\":\"w\",\"class\":\"bad\",\"values\":[2,3]}\n",
            "{\"tensor\":\"w\",\"class\":\"good\",\"values\":[1]}\n{\"tensor\":\"w\",\"class\":\"bad\",\"values\":[1]}\n",
            "{\"tensor\":\"w\",\"class\":\"invalid\",\"values\":[1]}\n"
        ] {
            let (root,path)=setup(value);
            assert!(compile_activation_recipe(&path,1.0,false).is_err());
            fs::remove_dir_all(root).unwrap();
        }
    }
}
