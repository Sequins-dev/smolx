import Testing

@testable import smolx

@Suite("ToolSchemaNormalizer")
struct ToolSchemaNormalizerTests {
    @Test func nullableTypeUnionBecomesStringTypeWithNullableFlag() {
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
            "type": .object([
                "properties": .object([
                    "path": .object([
                        "description": .string("Path to inspect"),
                        "nullable": .bool(true),
                        "type": .string("string"),
                    ]),
                ]),
                "required": .array([]),
                "type": .string("object"),
            ]),
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
            "type": .object([
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
                        "type": .string("object"),
                    ]),
                ]),
                "type": .string("object"),
            ]),
        ]))
    }
}
