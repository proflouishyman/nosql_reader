.PHONY: stamp-build-info pre-merge pre-release

stamp-build-info:
	./scripts/stamp_build_info.sh

pre-merge: stamp-build-info
	@echo "Pre-merge checks: build metadata stamped."

pre-release: stamp-build-info
	@echo "Pre-release checks: build metadata stamped."
