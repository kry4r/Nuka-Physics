Write-Host "Building Nuka Physics Engine (Release)..."

cmake --build build --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit 1
}

Write-Host "Running tests..."

ctest --test-dir build -C Release --output-on-failure

if ($LASTEXITCODE -ne 0) {
    Write-Error "Some tests failed."
    exit 1
}

Write-Host "All tests passed."
