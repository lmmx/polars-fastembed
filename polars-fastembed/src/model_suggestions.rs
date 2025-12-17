// src/model_suggestions.rs

use std::collections::{HashSet, BTreeMap};
use std::cmp::Ordering;

use fastembed::{TextEmbedding, EmbeddingModel, ModelInfo};
use polars::prelude::{PolarsError, PolarsResult};

/// Tokenize by splitting on dash/underscore/dot and lowercasing.
fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| c == '-' || c == '_' || c == '.')
        .map(|part| part.to_lowercase())
        .collect()
}

/// Naive Hamming distance (pairwise char mismatches + length difference).
fn hamming_distance(a: &str, b: &str) -> usize {
    let mismatches = a.chars().zip(b.chars()).filter(|(x, y)| x != y).count();
    mismatches + a.len().abs_diff(b.len())
}

/// Convert an EmbeddingModel enum variant to its string name.
fn model_variant_name(model: &EmbeddingModel) -> String {
    format!("{:?}", model)
}

/// Locate a matching model by enum variant name, full `model_code`, or produce suggestions
/// with a detailed list of all known models grouped by dimension.
pub fn from_model_code(code: &str) -> PolarsResult<EmbeddingModel> {
    // The "suffix" after the slash (if any). E.g. "BAAI/bge-small-en-v1.5" -> "bge-small-en-v1.5"
    let user_suffix = code
        .rsplit_once('/')
        .map(|(_, suffix)| suffix)
        .unwrap_or(code);

    // All known text-embedding models
    let all_models = TextEmbedding::list_supported_models();

    // 1) Check for exact match on enum variant name (case-insensitive)
    let code_lower = code.to_lowercase();
    if let Some(exact) = all_models.iter().find(|m| {
        model_variant_name(&m.model).to_lowercase() == code_lower
    }) {
        return Ok(exact.model.clone());
    }

    // 2) Check for exact match on model_code (HuggingFace repo path)
    if let Some(exact) = all_models.iter().find(|m| m.model_code == code) {
        return Ok(exact.model.clone());
    }

    // 3) Otherwise, build a suggestion list
    let user_tokens: HashSet<_> = tokenize(user_suffix).into_iter().collect();
    let mut scored_suggestions: Vec<(i32, usize, &ModelInfo<EmbeddingModel>)> = all_models
        .iter()
        .map(|info| {
            let variant_name = model_variant_name(&info.model);
            let full_code = info.model_code.as_str();
            let suffix = full_code.rsplit_once('/').map(|(_, s)| s).unwrap_or(full_code);

            // Combine tokens from both variant name and model_code suffix
            let mut candidate_tokens: HashSet<_> = tokenize(suffix).into_iter().collect();
            candidate_tokens.extend(tokenize(&variant_name));

            // Score #1: number of matching tokens
            let match_count = user_tokens.intersection(&candidate_tokens).count() as i32;

            // Score #2: tie-break via hamming distance (lower is better)
            // Check distance against both variant name and suffix
            let distance_suffix = hamming_distance(user_suffix, suffix);
            let distance_variant = hamming_distance(&code_lower, &variant_name.to_lowercase());
            let distance = distance_suffix.min(distance_variant);

            // We store negative match_count so we can sort descending
            (-match_count, distance, info)
        })
        .collect();

    // Sort by (descending match_count, ascending hamming distance, alphabetical code)
    scored_suggestions.sort_by(|a, b| {
        let (a_neg_match, a_dist, a_info) = a;
        let (b_neg_match, b_dist, b_info) = b;

        // descending match_count => ascending -match_count
        let match_cmp = a_neg_match.cmp(b_neg_match);
        if match_cmp != Ordering::Equal {
            return match_cmp;
        }

        // tie-break by ascending distance
        let dist_cmp = a_dist.cmp(b_dist);
        if dist_cmp != Ordering::Equal {
            return dist_cmp;
        }

        // final tie-break by model_code
        a_info.model_code.cmp(&b_info.model_code)
    });

    // Grab just the top 5 for "Did you mean" block
    let top_suggestions = scored_suggestions
        .iter()
        .take(5)
        .enumerate()
        .map(|(i, &(_, _, info))| {
            let variant = model_variant_name(&info.model);
            format!("  {}. {} ({})", i + 1, variant, info.model_code)
        })
        .collect::<Vec<_>>()
        .join("\n");

    // 4) Build a grouped listing of ALL models by dimension
    let mut by_dim: BTreeMap<u32, Vec<&ModelInfo<EmbeddingModel>>> = BTreeMap::new();
    for info in &all_models {
        by_dim.entry(info.dim.try_into().unwrap()).or_default().push(info);
    }

    // For each dimension, sort by model_code, then add lines
    let mut dimension_blocks = Vec::with_capacity(by_dim.len());
    for (dim, infos) in &by_dim {
        let mut lines = Vec::new();
        lines.push(format!("Dimension: {}", dim));

        let mut sorted_infos = infos.clone();
        sorted_infos.sort_by_key(|i| i.model_code.as_str());

        for i in sorted_infos {
            let variant = model_variant_name(&i.model);
            lines.push(format!("  - {} ({})", variant, i.model_code));
            lines.push(format!("    \"{}\"", i.description));
        }
        dimension_blocks.push(lines.join("\n"));
    }

    let grouped_listing = dimension_blocks.join("\n\n");

    // 5) Return a Polars error with suggestions + the full grouped listing
    Err(PolarsError::ComputeError(
        format!(
            "Unsupported or unknown model: {code}\n\n\
             Did you mean one of:\n\
             {top_suggestions}\n\
             \n\
             All known models, grouped by dimension:\n\
             {grouped_listing}"
        )
        .into(),
    ))
}
