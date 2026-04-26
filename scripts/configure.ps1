param(
    [switch]$Tests = $true
)

$testsFlag = if ($Tests) { "ON" } else { "OFF" }

Write-Host "Configuring Nuka Physics Engine..."
Write-Host "  Tests: $testsFlag"

cmake -S . -B build -DNK_BUILD_TESTS:BOOL=$testsFlag

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed."
    exit 1
}

Write-Host "Configuration complete. Build directory: build/"
