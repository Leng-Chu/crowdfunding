$ErrorActionPreference = 'Stop'

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $paperDir

try {
    New-Item -ItemType Directory -Force out | Out-Null
    xelatex -output-directory=out -interaction=nonstopmode -halt-on-error main.tex | Out-Null
    xelatex -output-directory=out -interaction=nonstopmode -halt-on-error main.tex | Out-Null

    @(
        'main.aux',
        'main.lof',
        'main.log',
        'main.lot',
        'main.out',
        'main.pdf',
        'main.toc'
    ) | ForEach-Object {
        Remove-Item -LiteralPath $_ -Force -ErrorAction SilentlyContinue
    }
}
finally {
    Pop-Location
}
