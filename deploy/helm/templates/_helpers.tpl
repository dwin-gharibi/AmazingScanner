{{/* Name helpers, following the conventions `helm create` establishes. */}}

{{- define "amazingscanner.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "amazingscanner.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "amazingscanner.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "amazingscanner.labels" -}}
helm.sh/chart: {{ include "amazingscanner.chart" . }}
{{ include "amazingscanner.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: amazingscanner
{{- end -}}

{{/*
Selector labels must never include anything that changes on upgrade — a
Deployment's selector is immutable, so putting the chart or app version in here
makes every chart bump a manual delete-and-recreate.
*/}}
{{- define "amazingscanner.selectorLabels" -}}
app.kubernetes.io/name: {{ include "amazingscanner.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "amazingscanner.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "amazingscanner.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "amazingscanner.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}
