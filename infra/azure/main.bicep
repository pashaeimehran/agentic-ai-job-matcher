// Job Match Agent on Azure Container Apps + Azure Database for PostgreSQL Flexible Server (pgvector).
// Low-cost portfolio settings: Burstable B1ms PostgreSQL, Basic ACR, consumption Container Apps
// scaled 0..1. Deployed in two passes by deploy.sh: deployApp=false (platform), then a Docker build and
// push to the registry, then deployApp=true (Container App).
targetScope = 'resourceGroup'

@description('Short lowercase prefix for resource names (3-12 chars).')
@minLength(3)
@maxLength(12)
param prefix string = 'jobmatch'

param location string = resourceGroup().location

@description('false: platform only (registry, environment, database). true: also the Container App.')
param deployApp bool = false

@description('Image tag in the registry, e.g. a commit SHA.')
param imageTag string = 'latest'

@allowed(['fixture', 'live'])
param appMode string = 'fixture'

param openAiModel string = 'gpt-4o-mini'

@description('PostgreSQL administrator login name.')
param postgresAdminLogin string = 'jobmatchadmin'

@secure()
@description('PostgreSQL administrator password (from the environment, never committed).')
param postgresAdminPassword string

@secure()
@description('OpenAI API key; required only when appMode is live.')
param openAiApiKey string = ''

@description('Optional Langfuse host (e.g. Langfuse Cloud EU). Empty disables tracing.')
param langfuseHost string = ''
param langfusePublicKey string = ''
@secure()
param langfuseSecretKey string = ''

var suffix = uniqueString(resourceGroup().id)
var registryName = toLower('${prefix}acr${suffix}')
var postgresName = toLower('${prefix}-pg-${suffix}')
var databaseName = 'careermatch'
var tracing = !empty(langfuseHost) && !empty(langfusePublicKey) && !empty(langfuseSecretKey)
var databaseUrl = 'postgresql://${postgresAdminLogin}:${uriComponent(postgresAdminPassword)}@${postgres.properties.fullyQualifiedDomainName}:5432/${databaseName}?sslmode=require'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs-${suffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

// The app pulls images with a managed identity: no registry password exists anywhere.
resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-pull-${suffix}'
  location: location
}

var acrPullRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, pullIdentity.id, acrPullRole)
  scope: registry
  properties: {
    roleDefinitionId: acrPullRole
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env-${suffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2023-12-01-preview' = {
  name: postgresName
  location: location
  sku: { name: 'Standard_B1ms', tier: 'Burstable' }
  properties: {
    version: '16'
    administratorLogin: postgresAdminLogin
    administratorLoginPassword: postgresAdminPassword
    storage: { storageSizeGB: 32, autoGrow: 'Disabled' }
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
    network: { publicNetworkAccess: 'Enabled' }
    authConfig: { activeDirectoryAuth: 'Disabled', passwordAuth: 'Enabled' }
  }
}

// pgvector must be allow-listed before CREATE EXTENSION vector (run by the app's schema init).
resource vectorExtension 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2023-12-01-preview' = {
  parent: postgres
  name: 'azure.extensions'
  properties: { value: 'VECTOR', source: 'user-override' }
}

resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-12-01-preview' = {
  parent: postgres
  name: databaseName
  properties: { charset: 'UTF8', collation: 'en_US.utf8' }
  dependsOn: [vectorExtension]
}

// Portfolio networking: public endpoint, TLS required, reachable from Azure-hosted clients only
// (the 0.0.0.0 rule admits any Azure service, not only this subscription). Production would use
// VNet integration and private access instead (see docs/deployment.md).
// Flexible Server rejects concurrent child operations, so the children are applied one at a time.
resource allowAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-12-01-preview' = {
  parent: postgres
  name: 'AllowAzureServices'
  properties: { startIpAddress: '0.0.0.0', endIpAddress: '0.0.0.0' }
  dependsOn: [database]
}

var baseSecrets = [
  { name: 'database-url', value: databaseUrl }
]
var optionalSecrets = concat(
  empty(openAiApiKey) ? [] : [{ name: 'openai-api-key', value: openAiApiKey }],
  tracing ? [{ name: 'langfuse-secret-key', value: langfuseSecretKey }] : []
)
var baseEnv = [
  { name: 'APP_MODE', value: appMode }
  { name: 'APP_ENV', value: 'production' }
  { name: 'DATABASE_URL', secretRef: 'database-url' }
  { name: 'CV_ID', value: 'synthetic-demo-${appMode}' }
  { name: 'OPENAI_MODEL', value: openAiModel }
  { name: 'AGENT_PROMPT_VERSION', value: 'v2' }
  { name: 'AGENT_MAX_RUN_TOKENS', value: '60000' }
  { name: 'FORWARDED_ALLOW_IPS', value: '*' }
  { name: 'TRACING_ENABLED', value: tracing ? 'true' : 'false' }
]
var optionalEnv = concat(
  empty(openAiApiKey) ? [] : [{ name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }],
  tracing ? [
    { name: 'LANGFUSE_HOST', value: langfuseHost }
    { name: 'LANGFUSE_PUBLIC_KEY', value: langfusePublicKey }
    { name: 'LANGFUSE_SECRET_KEY', secretRef: 'langfuse-secret-key' }
  ] : []
)

resource app 'Microsoft.App/containerApps@2024-03-01' = if (deployApp) {
  name: '${prefix}-api'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${pullIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [{ server: registry.properties.loginServer, identity: pullIdentity.id }]
      secrets: concat(baseSecrets, optionalSecrets)
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${registry.properties.loginServer}/job-match-agent:${imageTag}'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat(baseEnv, optionalEnv)
          probes: [
            { type: 'Startup', httpGet: { path: '/health', port: 8000 }, periodSeconds: 5, failureThreshold: 24 }
            { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, periodSeconds: 30 }
            { type: 'Readiness', httpGet: { path: '/ready', port: 8000 }, periodSeconds: 15, failureThreshold: 3 }
          ]
        }
      ]
      // One replica at most: conversation threads and the response cache are process-local.
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
  dependsOn: [acrPull, database, allowAzure]
}

output registryName string = registry.name
output registryLoginServer string = registry.properties.loginServer
output postgresServer string = postgres.properties.fullyQualifiedDomainName
output appUrl string = deployApp ? 'https://${app!.properties.configuration.ingress.fqdn}' : ''
