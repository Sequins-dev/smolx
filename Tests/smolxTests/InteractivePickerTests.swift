import Testing
import Foundation
@testable import smolx

@Suite("InteractivePicker")
struct InteractivePickerTests {

    // MARK: - truncate

    @Test func truncateLeavesShortStringsAlone() {
        #expect(InteractivePicker.truncate("hello", to: 10) == "hello")
        #expect(InteractivePicker.truncate("hello", to: 5) == "hello")
    }

    @Test func truncateAddsEllipsisWhenTooLong() {
        // 14 chars → keep 7 + ellipsis = 8 total.
        #expect(InteractivePicker.truncate("Qwen3.6-27B-4bit", to: 8) == "Qwen3.6…")
    }

    @Test func truncateEdgeCases() {
        #expect(InteractivePicker.truncate("abc", to: 0) == "")
        #expect(InteractivePicker.truncate("abc", to: 1) == "…")
        // Exactly at the boundary returns the original string, no ellipsis.
        #expect(InteractivePicker.truncate("abc", to: 3) == "abc")
        #expect(InteractivePicker.truncate("abcd", to: 3) == "ab…")
    }

    // MARK: - marqueeFrame

    @Test func marqueeFramePadsShortStringsToWidth() {
        // Short string just gets right-padded; padding is essential so
        // the row doesn't shrink and leave stale bytes from the prior
        // frame visible on the right.
        let out = InteractivePicker.marqueeFrame("hi", offset: 5, width: 6)
        #expect(out == "hi    ")
        #expect(out.count == 6)
    }

    @Test func marqueeFrameAtOffsetZeroShowsLeadingEdge() {
        // offset=0 should show the first `width` chars of the string —
        // the natural "starts here" frame the user sees right after a
        // selection change.
        let s = "abcdefghij"
        let out = InteractivePicker.marqueeFrame(s, offset: 0, width: 4)
        #expect(out == "abcd")
        #expect(out.count == 4)
    }

    @Test func marqueeFrameScrollsLeftAsOffsetIncreases() {
        let s = "abcdefghij"
        #expect(InteractivePicker.marqueeFrame(s, offset: 1, width: 4) == "bcde")
        #expect(InteractivePicker.marqueeFrame(s, offset: 2, width: 4) == "cdef")
        #expect(InteractivePicker.marqueeFrame(s, offset: 3, width: 4) == "defg")
    }

    @Test func marqueeFrameWrapsThroughSeparatorAndRestartsAtZero() {
        // After scrolling past the full string + separator, the buffer
        // should wrap back to the leading "abc…" frame. The period is
        // s.count + separator.count = 10 + 7 = 17.
        let s = "abcdefghij"
        let frame0 = InteractivePicker.marqueeFrame(s, offset: 0, width: 4)
        let frame17 = InteractivePicker.marqueeFrame(s, offset: 17, width: 4)
        #expect(frame0 == frame17)
    }

    @Test func marqueeFrameHandlesNegativeOffsetDefensively() {
        // Floor-mod (positive remainder) regardless of input sign.
        let s = "abcdefghij"
        let f0 = InteractivePicker.marqueeFrame(s, offset: 0, width: 4)
        let fNeg = InteractivePicker.marqueeFrame(s, offset: -17, width: 4)
        #expect(f0 == fNeg)
    }

    // MARK: - viewport

    @Test func viewportTopUnchangedWhenSelectionInside() {
        // selected=2 fits in [1, 5) so top stays at 1.
        #expect(InteractivePicker.viewport(
            selected: 2, total: 20, top: 1, size: 4) == 1)
    }

    @Test func viewportScrollsDownToKeepSelectionVisible() {
        // selected=5 is past viewport [1, 5); top must shift to 2 so
        // selection lands at the bottom of [2, 6).
        #expect(InteractivePicker.viewport(
            selected: 5, total: 20, top: 1, size: 4) == 2)
    }

    @Test func viewportScrollsUpToKeepSelectionVisible() {
        // selected=0 is before viewport [3, 7); top snaps to 0.
        #expect(InteractivePicker.viewport(
            selected: 0, total: 20, top: 3, size: 4) == 0)
    }

    @Test func viewportClampsToMaxTopWhenNearEnd() {
        // total=10, size=4 → maxTop=6. selected at the end should not
        // push top past 6 (would render past the array).
        #expect(InteractivePicker.viewport(
            selected: 9, total: 10, top: 8, size: 4) == 6)
    }

    @Test func viewportHandlesEmptyAndZeroSize() {
        #expect(InteractivePicker.viewport(
            selected: 0, total: 0, top: 0, size: 4) == 0)
        #expect(InteractivePicker.viewport(
            selected: 5, total: 10, top: 3, size: 0) == 0)
    }
}
