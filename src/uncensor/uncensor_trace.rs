//! One-command checkpoint editing from already captured, paired activations.
//! Collection, refusal evaluation, and strength search remain separate stages.

use crate::activation_trace::{compile_activation_recipe, ActivationRecipe};
use crate::model_edit::{
    prepare_model_abliteration_value, run_model_abliteration_with_recipe, ModelAblationPlan,
};
use serde_json::{json, Value};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

fn invalid(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

pub struct UncensorTracePlan {
    trace: PathBuf,
    strength: f64,
    preserve_row_norms: bool,
    activation: ActivationRecipe,
    model: ModelAblationPlan,
}

impl UncensorTracePlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.uncensor_trace_plan.v1",
            "trace":self.trace.to_string_lossy(),
            "activation_records":self.activation.records,
            "activation_tensors":self.activation.tensors,
            "strength":self.strength,
            "preserve_row_norms":self.preserve_row_norms,
            "model_edit":self.model.as_json(),
            "activation_capture":"EXTERNAL_INSTRUMENTED_RUNTIME",
            "strength_optimization":"NOT_RUN",
            "refusal_rate":"NOT_EVALUATED",
            "capability_retention":"NOT_EVALUATED",
            "status":"PLAN_ONLY"
        })
    }
}

pub fn prepare_uncensor_from_trace(
    source: &Path,
    trace: &Path,
    output: &Path,
    strength: f64,
    preserve_row_norms: bool,
) -> io::Result<UncensorTracePlan> {
    let trace = trace.canonicalize()?;
    let activation = compile_activation_recipe(&trace, strength, preserve_row_norms)?;
    let model = prepare_model_abliteration_value(source, &activation.recipe, output)?;
    Ok(UncensorTracePlan {
        trace,
        strength,
        preserve_row_norms,
        activation,
        model,
    })
}

pub fn run_uncensor_from_trace(
    plan: &UncensorTracePlan,
    mut log: impl FnMut(&str),
) -> io::Result<Value> {
    let current = compile_activation_recipe(&plan.trace, plan.strength, plan.preserve_row_norms)?;
    if current.records != plan.activation.records || current.recipe != plan.activation.recipe {
        return Err(invalid("activation trace changed after planning"));
    }
    let result =
        run_model_abliteration_with_recipe(&plan.model, &plan.activation.recipe, |message| {
            log(message)
        })?;
    Ok(json!({
        "schema":"ojon.uncensor_trace_result.v1",
        "model":result.as_json(),
        "activation_records":plan.activation.records,
        "activation_tensors":plan.activation.tensors,
        "recipe":result.output.join("ojon-activation-recipe.json").to_string_lossy(),
        "original_checkpoint":"PRESERVED",
        "activation_capture":"EXTERNAL_INSTRUMENTED_RUNTIME",
        "strength_optimization":"NOT_RUN",
        "model_load":"NOT_TESTED",
        "refusal_rate":"NOT_EVALUATED",
        "capability_retention":"NOT_EVALUATED",
        "status":"EDITED_TEACHER_NOT_VALIDATED"
    }))
}
