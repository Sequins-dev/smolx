enum ToolSchemaNormalizer {
    static func normalize(_ value: JSONValue) -> JSONValue {
        switch value {
        case .array(let values):
            return .array(values.map(normalize))
        case .object(let object):
            if object.count == 1, case .object(let wrapped)? = object["type"] {
                return .object(normalizeSchemaObject(wrapped))
            }
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

        if let type = normalized["type"] {
            let typeInfo = normalizeType(type)
            normalized["type"] = .string(typeInfo.type)
            if typeInfo.nullable {
                normalized["nullable"] = .bool(true)
            }
        } else {
            normalized["type"] = inferredType(for: normalized)
        }

        return normalized
    }

    private static func normalizeType(_ value: JSONValue) -> (type: String, nullable: Bool) {
        switch value {
        case .string(let string):
            return (string, false)
        case .array(let values):
            let stringTypes = values.compactMap { value -> String? in
                if case .string(let string) = value { return string }
                return nil
            }
            let hasNull = values.contains { value in
                isNullType(value)
            }
            return (
                stringTypes.first(where: { $0.lowercased() != "null" }) ?? "string",
                hasNull)
        case .object(let object):
            let normalized = normalizeSchemaObject(object)
            if case .string(let type)? = normalized["type"] {
                let nullable = normalized["nullable"] == .bool(true)
                return (type, nullable)
            }
            return ("string", false)
        case .null:
            return ("string", true)
        default:
            return ("string", false)
        }
    }

    private static func isNullType(_ value: JSONValue) -> Bool {
        switch value {
        case .string(let string):
            return string.lowercased() == "null"
        case .null:
            return true
        default:
            return false
        }
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
