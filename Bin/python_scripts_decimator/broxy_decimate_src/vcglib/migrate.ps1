# Usage: run from inside the vcglib folder
$root = Get-Location

# 1) Replace all includes that reference Eigen -> Eigen
$files = Get-ChildItem -Path $root -Recurse -File |
  Where-Object { $_.FullName -notmatch "\\\.git\\" } |
  Select-String -Pattern "Eigen" -List |
  Select-Object -ExpandProperty Path -Unique

foreach ($f in $files) {
  (Get-Content $f) `
    -replace "Eigen/", "Eigen/" `
    -replace "Eigen", "Eigen" |
    Set-Content $f
}

# 2) Fix vcg/math/eigen.h to use system includes
$eigenHeader = Join-Path $root "vcg/math/eigen.h"
if (Test-Path $eigenHeader) {
  (Get-Content $eigenHeader) `
    -replace '#include\s+"..\/..\/Eigen\/LU"', '#include <Eigen/LU>' `
    -replace '#include\s+"..\/..\/Eigen\/Geometry"', '#include <Eigen/Geometry>' `
    -replace '#include\s+"..\/..\/Eigen\/Array"', '#include <Eigen/Array>' `
    -replace '#include\s+"..\/..\/Eigen\/Core"', '#include <Eigen/Core>' |
    Set-Content $eigenHeader
}
