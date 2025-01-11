$secrets = $env:SECRETS_CONTEXT | ConvertFrom-Json

$envFilePath = "$env:GITHUB_ENV"

foreach ($key in $secrets.PSObject.Properties.Name) {
    $value = $secrets.$key -replace '\r','' -replace '\n','' 
    Write-Host "Exporting environment variable for: $key"
    
    $escapedValue = $value -replace '"', '\"'

    "$key=$escapedValue" | Out-File -FilePath $envFilePath -Append
}
