import Foundation
import Testing

@testable import smolx

@Suite("ServeOptionParser")
struct ServeOptionParserTests {
    @Test func parseBytesSupportsBinarySuffixes() {
        #expect(ServeOptionParser.parseBytes("32GB") == Int64(32) * 1_073_741_824)
        #expect(ServeOptionParser.parseBytes("512MB") == Int64(512) * 1_048_576)
        #expect(ServeOptionParser.parseBytes("1.5G") == Int64(1.5 * 1_073_741_824))
        #expect(ServeOptionParser.parseBytes("2048") == 2048)
    }

    @Test func parseBytesRejectsMalformedInput() {
        #expect(ServeOptionParser.parseBytes("oops") == nil)
        #expect(ServeOptionParser.parseBytes("1XB") == nil)
    }

    @Test func parseDurationReturnsSeconds() {
        #expect(ServeOptionParser.parseDuration("30s") == 30)
        #expect(ServeOptionParser.parseDuration("10m") == 600)
        #expect(ServeOptionParser.parseDuration("1.5h") == 5400)
        #expect(ServeOptionParser.parseDuration("45") == 45)
    }
}
