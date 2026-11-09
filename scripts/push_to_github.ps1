$ErrorActionPreference = 'Stop'
$repo = 'https://github.com/x-kzy/UTS-DEEP-LEARNING.git'
$source = Split-Path -Parent $PSScriptRoot
$nonce = [guid]::NewGuid().ToString('N').Substring(0, 8)
$working = Join-Path $env:TEMP ("uts_dl_repo_" + $nonce)
$dest = Join-Path $working 'repository'
$items = @('README.md','.gitignore','requirements.txt','THIRD_PARTY_NOTICES.md','PANDUAN_UPLOAD_GITHUB.txt','PUSH_KE_GITHUB.cmd','notebooks','reports','src','data','docs','results','web_static','scripts','config','.github')

Write-Host 'Repository tujuan:' $repo
Write-Host 'Paket sumber:' $source
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git belum dipasang.' }
New-Item -ItemType Directory -Path $working -Force | Out-Null
try {
    git clone --depth 1 --branch main $repo $dest
    if ($LASTEXITCODE -ne 0) { throw 'Gagal melakukan clone. Periksa koneksi atau hak akses.' }
    foreach ($name in $items) {
        $a = Join-Path $source $name
        $b = Join-Path $dest $name
        if (-not (Test-Path -LiteralPath $a)) { throw ("File paket hilang: " + $a) }
        if (Test-Path -LiteralPath $a -PathType Container) {
            New-Item -ItemType Directory -Path $b -Force | Out-Null
            Copy-Item -Path (Join-Path $a '*') -Destination $b -Force -Recurse -ErrorAction Stop
        } else {
            Copy-Item -LiteralPath $a -Destination $b -Force -ErrorAction Stop
        }
    }
    Push-Location $dest
    try {
        git add -- README.md .gitignore requirements.txt THIRD_PARTY_NOTICES.md PANDUAN_UPLOAD_GITHUB.txt PUSH_KE_GITHUB.cmd notebooks reports src data docs results web_static scripts config .github
        if ($LASTEXITCODE -ne 0) { throw 'Git add gagal.' }
        Write-Host "`n==== RINGKASAN PERUBAHAN ====" -ForegroundColor Cyan
        git diff --cached --stat
        git status --short
        git diff --cached --quiet
        if ($LASTEXITCODE -eq 0) { Write-Host 'Tidak ada perubahan untuk dikirim.'; exit 0 }
        Write-Host "`nLICENSE dan file lama lainnya dipertahankan. Tidak ada force push." -ForegroundColor Yellow
        $answer = Read-Host 'Ketik YA untuk commit dan push ke main (selain YA = batal)'
        if ($answer -cne 'YA') { Write-Host 'Dibatalkan. Tidak ada perubahan di remote.'; exit 1 }
        git commit -m 'docs: publish UTS Deep Learning Kelompok 4 experiments and Spaces demo'
        if ($LASTEXITCODE -ne 0) { throw 'Commit gagal. Periksa git user.name/user.email di komputer.' }
        git push origin main
        if ($LASTEXITCODE -ne 0) { throw 'Push ditolak. Periksa login GitHub/hak akses, atau perubahan terbaru branch main.' }
        Write-Host "`nBERHASIL: https://github.com/x-kzy/UTS-DEEP-LEARNING" -ForegroundColor Green
    } finally { Pop-Location }
} finally {
  Write-Host "Folder kerja sementara: $working"
  # Deliberately do not delete: on failed push, it contains a recoverable local commit.
}
