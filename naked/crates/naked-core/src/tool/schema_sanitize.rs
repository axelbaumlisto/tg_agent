//! MCP tool schema sanitizer — fix dirty schemas before sending to providers.
//!
//! Handles: nullable unions, bare objects, dangling required, single-element unions.

use serde_json::{Map, Value};

/// Sanitize a JSON Schema in-place (recursive).
pub fn sanitize(schema: &mut Value) {
    collapse_nullable(schema);
    inject_bare_properties(schema);
    prune_dangling_required(schema);
    collapse_single_union(schema);
    // Recurse into sub-schemas:
    if let Some(obj) = schema.as_object_mut() {
        for v in obj.values_mut() {
            sanitize(v);
        }
    } else if let Some(arr) = schema.as_array_mut() {
        for v in arr {
            sanitize(v);
        }
    }
}

/// `{"anyOf":[X, {"type":"null"}]}` → `X ∪ {"nullable": true}`
fn collapse_nullable(schema: &mut Value) {
    let any_of = match schema
        .as_object()
        .and_then(|o| o.get("anyOf"))
        .and_then(|v| v.as_array())
    {
        Some(a) if a.len() == 2 => a.clone(),
        _ => return,
    };
    let null_idx = any_of
        .iter()
        .position(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"));
    let Some(null_idx) = null_idx else { return };
    let real_idx = 1 - null_idx;
    let mut real = any_of[real_idx].clone();
    if let Some(obj) = real.as_object_mut() {
        obj.insert("nullable".into(), Value::Bool(true));
    }
    if let Some(outer) = schema.as_object_mut() {
        outer.remove("anyOf");
    }
    // Merge real into schema:
    if let (Some(outer), Some(inner)) = (schema.as_object_mut(), real.as_object()) {
        for (k, v) in inner {
            outer.insert(k.clone(), v.clone());
        }
    }
}

/// `{"type":"object"}` without "properties" → inject empty properties.
fn inject_bare_properties(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut()
        && obj.get("type").and_then(|t| t.as_str()) == Some("object")
        && !obj.contains_key("properties")
    {
        obj.insert("properties".into(), Value::Object(Map::new()));
    }
}

/// Prune `required` entries not present in `properties`.
fn prune_dangling_required(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    let props: std::collections::HashSet<String> = obj
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|p| p.keys().cloned().collect())
        .unwrap_or_default();
    if let Some(req) = obj.get_mut("required").and_then(|v| v.as_array_mut()) {
        req.retain(|v| v.as_str().is_some_and(|s| props.contains(s)));
    }
}

/// `{"oneOf":[X]}` or `{"allOf":[X]}` → unwrap to X.
fn collapse_single_union(schema: &mut Value) {
    for key in ["oneOf", "allOf"] {
        let single = schema
            .as_object()
            .and_then(|o| o.get(key))
            .and_then(|v| v.as_array())
            .filter(|a| a.len() == 1)
            .map(|a| a[0].clone());
        if let Some(inner) = single {
            if let Some(obj) = schema.as_object_mut() {
                obj.remove(key);
            }
            if let (Some(outer), Some(inner_obj)) = (schema.as_object_mut(), inner.as_object()) {
                for (k, v) in inner_obj {
                    outer.insert(k.clone(), v.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nullable_union() {
        let mut s = json!({"anyOf": [{"type": "string"}, {"type": "null"}]});
        sanitize(&mut s);
        assert_eq!(s["type"], "string");
        assert_eq!(s["nullable"], true);
    }

    #[test]
    fn bare_object() {
        let mut s = json!({"type": "object"});
        sanitize(&mut s);
        assert!(s["properties"].is_object());
    }

    #[test]
    fn dangling_required() {
        let mut s = json!({"type": "object", "properties": {"a": {"type":"string"}}, "required": ["a","b"]});
        sanitize(&mut s);
        assert_eq!(s["required"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn single_oneof() {
        let mut s = json!({"oneOf": [{"type": "integer"}]});
        sanitize(&mut s);
        assert_eq!(s["type"], "integer");
    }

    #[test]
    fn nested() {
        let mut s = json!({"type": "object", "properties": {"x": {"anyOf": [{"type":"number"}, {"type":"null"}]}}});
        sanitize(&mut s);
        assert_eq!(s["properties"]["x"]["type"], "number");
        assert_eq!(s["properties"]["x"]["nullable"], true);
    }
}
