enum ToolSchemaNormalizer {
    static func normalize(_ value: JSONValue) -> JSONValue {
        switch value {
        case .array(let values):
            return .array(values.map(normalize))
        case .object(let object):
            return .object(normalizeSchemaObject(object))
        default:
            return value
        }
    }

    private static func normalizeSchemaObject(_ object: [String: JSONValue]) -> [String: JSONValue] {
        var normalized: [String: JSONValue] = [:]
        for (key, value) in object {
            normalized[key] = normalizeValue(value, forKey: key)
        }

        if normalized["type"] == nil {
            normalized["type"] = inferredType(for: normalized)
        }

        guard case .array(let typeValues)? = normalized["type"] else { return normalized }

        let stringTypes = typeValues.compactMap { value -> String? in
            if case .string(let string) = value { return string }
            return nil
        }
        let hasNull = typeValues.contains { value in
            if case .string(let string) = value { return string.lowercased() == "null" }
            if case .null = value { return true }
            return false
        }

        if let primaryType = stringTypes.first(where: { $0.lowercased() != "null" }) {
            normalized["type"] = .string(primaryType)
        }
        if hasNull {
            normalized["nullable"] = .bool(true)
        }

        return normalized
    }

    private static func normalizeValue(_ value: JSONValue, forKey key: String) -> JSONValue {
        switch (key, value) {
        case ("properties", .object(let properties)):
            return .object(properties.mapValues(normalize))
        case ("additionalProperties", .object(let schema)) where schema.isEmpty:
            return value
        default:
            return normalize(value)
        }
    }

    private static func inferredType(for object: [String: JSONValue]) -> JSONValue? {
        guard !object.isEmpty else { return nil }
        if object["properties"] != nil || object["additionalProperties"] != nil {
            return .string("object")
        }
        if object["items"] != nil {
            return .string("array")
        }
        if object["enum"] != nil || object["const"] != nil || object["description"] != nil {
            return .string("string")
        }
        return .string("string")
    }
}
