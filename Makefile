CLT_FRAMEWORKS := /Library/Developer/CommandLineTools/Library/Developer/Frameworks
CLT_LIBS       := /Library/Developer/CommandLineTools/Library/Developer/usr/lib

# Extra flags so `swift test` finds Testing.framework + lib_TestingInterop.dylib
# under Command Line Tools (full-Xcode installs don't need these but harmless).
TEST_FLAGS := \
	-Xswiftc -F -Xswiftc $(CLT_FRAMEWORKS) \
	-Xlinker -rpath -Xlinker $(CLT_FRAMEWORKS) \
	-Xlinker -rpath -Xlinker $(CLT_LIBS)

XCODE_PROJECT := smolx.xcodeproj
XCODE_SCHEME  := smolx
XCODE_CONFIG  := Release
XCODE_BUILD_DIR := $(CURDIR)/build

.PHONY: build test run clean xcode xcode-build xcode-open install

# --- SwiftPM (compiles cleanly but the resulting binary cannot run MLX
# inference because Metal shaders aren't compiled by the CLI toolchain).
# Use this for tests and quick iteration on non-MLX code paths.

build:
	swift build

test:
	swift test $(TEST_FLAGS)

run:
	swift run smolx $(ARGS)

clean:
	rm -rf .build $(XCODE_PROJECT) $(XCODE_BUILD_DIR)

# --- Xcode build (required for MLX runtime — invokes `xcrun metal` to compile
# shaders into the `Cmlx` bundle). Requires full Xcode (or the standalone
# Metal Toolchain) installed and selected via `xcode-select -s`.

# Always regenerate the xcodeproj: xcodegen produces a static project file
# that doesn't auto-rediscover Swift files added under Sources/ afterwards.
# A make-style file-dependency on Sources/ doesn't reliably catch new files,
# so we just regenerate every time — cheap (<1s) and avoids silent staleness
# bugs where xcodebuild compiles an older snapshot of the source tree.
xcode:
	@command -v xcodegen >/dev/null || { echo "xcodegen not installed. Install with: brew install xcodegen"; exit 1; }
	@xcodegen generate >/dev/null

xcode-build: xcode
	@xcrun --find metal >/dev/null 2>&1 || { echo "xcrun metal not found. Install full Xcode (or the Metal Toolchain) and run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"; exit 1; }
	# mlx-swift-lm's MLXHuggingFaceMacros is a SwiftPM macro plugin; Xcode
	# defaults to refusing to load unverified macros on CLI builds. The two
	# skip flags green-light it without requiring an interactive trust prompt.
	xcodebuild \
		-project $(XCODE_PROJECT) \
		-scheme $(XCODE_SCHEME) \
		-configuration $(XCODE_CONFIG) \
		-derivedDataPath $(XCODE_BUILD_DIR) \
		-destination 'platform=macOS,arch=arm64' \
		-skipMacroValidation \
		-skipPackagePluginValidation \
		ONLY_ACTIVE_ARCH=YES \
		ARCHS=arm64 \
		build

xcode-open: xcode
	open $(XCODE_PROJECT)

# Copy the xcodebuild-produced binary AND its mlx-swift_Cmlx.bundle (which
# contains default.metallib) into the install dir. The binary alone won't work
# — MLX searches for the bundle as a sibling of the executable at runtime.
#
# After cp, we re-sign with `codesign -s -` (ad-hoc). The original signature
# from xcodebuild is still valid against the binary's bytes, BUT macOS Sonoma+
# attaches a `com.apple.provenance` xattr on cp that flips the kernel's
# execution-policy decision so the moved binary gets SIGKILL'd silently on
# launch. The xattr is system-protected and can't be removed; re-signing makes
# the kernel re-evaluate the binary as locally trusted from its new path.
INSTALL_DIR ?= $(HOME)/.local/bin
install: xcode-build
	@mkdir -p $(INSTALL_DIR)
	@PRODUCTS=$(XCODE_BUILD_DIR)/Build/Products/$(XCODE_CONFIG); \
		BIN="$$PRODUCTS/smolx"; \
		BUNDLE="$$PRODUCTS/mlx-swift_Cmlx.bundle"; \
		test -x "$$BIN"     || { echo "Missing built binary at $$BIN";          exit 1; }; \
		test -d "$$BUNDLE"  || { echo "Missing metallib bundle at $$BUNDLE";    exit 1; }; \
		cp     "$$BIN"        "$(INSTALL_DIR)/smolx"; \
		rm -rf "$(INSTALL_DIR)/mlx-swift_Cmlx.bundle"; \
		cp -R  "$$BUNDLE"     "$(INSTALL_DIR)/mlx-swift_Cmlx.bundle"; \
		codesign --force --sign - "$(INSTALL_DIR)/smolx" >/dev/null 2>&1; \
		echo "Installed: $(INSTALL_DIR)/smolx"; \
		echo "           $(INSTALL_DIR)/mlx-swift_Cmlx.bundle"
