import Testing

@testable import smolx

@Suite("ToolSchemaNormalizer")
struct ToolSchemaNormalizerTests {
    @Test func schemaWrappedInsideTypeKeyIsUnwrappedForGemmaTemplate() {
        let schema = JSONValue.object([
            "type": .object([
                "properties": .object([
                    "path": .object([
                        "description": .string("Path to inspect"),
                        "type": .array([.string("string"), .string("null")]),
                    ]),
                ]),
                "required": .array([]),
                "type": .string("object"),
            ]),
        ])

        #expect(ToolSchemaNormalizer.normalize(schema) == .object([
            "properties": .object([
                "path": .object([
                    "description": .string("Path to inspect"),
                    "nullable": .bool(true),
                    "type": .string("string"),
                ]),
            ]),
            "required": .array([]),
            "type": .string("object"),
        ]))
    }

    @Test func nullableTypeUnionBecomesStringTypeWithNullableFlag() {
        let schema = JSONValue.object([
            "properties": .object([
                "path": .object([
                    "description": .string("Path to inspect"),
                    "type": .array([.string("string"), .string("null")]),
                ]),
            ]),
            "required": .array([]),
            "type": .string("object"),
        ])

        #expect(ToolSchemaNormalizer.normalize(schema) == .object([
            "properties": .object([
                "path": .object([
                    "description": .string("Path to inspect"),
                    "nullable": .bool(true),
                    "type": .string("string"),
                ]),
            ]),
            "required": .array([]),
            "type": .string("object"),
        ]))
    }

    @Test func nestedTypeObjectsAreCollapsedToStringTypesForGemmaTemplate() {
        let schema = JSONValue.object([
            "properties": .object([
                "path": .object([
                    "description": .string("Path to inspect"),
                    "type": .object([
                        "type": .string("string")
                    ]),
                ]),
                "mode": .object([
                    "description": .string("Mode"),
                    "type": .array([.null]),
                ]),
                "enabled": .object([
                    "description": .string("Whether to enable"),
                    "type": .bool(true),
                ]),
            ]),
            "required": .array([]),
            "type": .string("object"),
        ])

        #expect(ToolSchemaNormalizer.normalize(schema) == .object([
            "properties": .object([
                "path": .object([
                    "description": .string("Path to inspect"),
                    "type": .string("string"),
                ]),
                "mode": .object([
                    "description": .string("Mode"),
                    "nullable": .bool(true),
                    "type": .string("string"),
                ]),
                "enabled": .object([
                    "description": .string("Whether to enable"),
                    "type": .string("string"),
                ]),
            ]),
            "required": .array([]),
            "type": .string("object"),
        ]))
    }

    @Test func booleanPropertySchemasAreWrappedForGemmaTemplate() {
        let schema = JSONValue.object([
            "properties": .object([
                "anything": .bool(true),
                "list": .object([
                    "items": .bool(true),
                    "type": .string("array"),
                ]),
            ]),
            "type": .string("object"),
        ])

        #expect(ToolSchemaNormalizer.normalize(schema) == .object([
            "properties": .object([
                "anything": .object([
                    "type": .string("string")
                ]),
                "list": .object([
                    "items": .object([
                        "type": .string("string")
                    ]),
                    "type": .string("array"),
                ]),
            ]),
            "type": .string("object"),
        ]))
    }

    @Test func objectAdditionalPropertiesDoesNotTriggerGemmaFallback() {
        let schema = JSONValue.object([
            "properties": .object([
                "metadata": .object([
                    "additionalProperties": .bool(true),
                    "type": .string("object"),
                ]),
            ]),
            "type": .string("object"),
        ])

        #expect(ToolSchemaNormalizer.normalize(schema) == .object([
            "properties": .object([
                "metadata": .object([
                    "additionalProperties": .bool(true),
                    "properties": .object([:]),
                    "type": .string("object"),
                ]),
            ]),
            "type": .string("object"),
        ]))
    }

    @Test func missingObjectAndArrayTypesAreInferredForGemmaTemplate() {
        let schema = JSONValue.object([
            "type": .object([
                "properties": .object([
                    "options": .object([
                        "items": .object([
                            "properties": .object([
                                "name": .object([
                                    "description": .string("Option name")
                                ])
                            ])
                        ])
                    ]),
                    "metadata": .object([
                        "additionalProperties": .object([:])
                    ]),
                ]),
                "type": .string("object"),
            ]),
        ])

        #expect(ToolSchemaNormalizer.normalize(schema) == .object([
            "properties": .object([
                "options": .object([
                    "items": .object([
                        "properties": .object([
                            "name": .object([
                                "description": .string("Option name"),
                                "type": .string("string"),
                            ])
                        ]),
                        "type": .string("object"),
                    ]),
                    "type": .string("array"),
                ]),
                "metadata": .object([
                    "additionalProperties": .object([:]),
                    "properties": .object([:]),
                    "type": .string("object"),
                ]),
            ]),
            "type": .string("object"),
        ]))
    }
}
