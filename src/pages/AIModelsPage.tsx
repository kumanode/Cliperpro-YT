import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, AudioLines, CheckCircle2, Cpu, ExternalLink, FolderOpen, HardDrive, KeyRound, Loader2, RefreshCw, Save, Sparkles, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  AI_PROVIDER_PRESETS,
  FALLBACK_MODELS,
  LOCAL_WHISPER_MODELS,
  presetForBaseUrl,
  signupLabel,
  TRANSCRIPTION_PRESETS,
  transcriptionPresetFor,
} from "@/config/aiProviders";
import {
  deleteLocalWhisperModel,
  listAIModels,
  probeLocalWhisper,
  validateLocalWhisperPath,
  type AIProviderSettings,
  type LocalWhisperPathValidation,
  type LocalWhisperProbe,
} from "@/hooks/appConfig";
import { useConfigStore } from "@/stores/configStore";
import { open as openUrl } from "@tauri-apps/plugin-shell";

export function AIModelsPage() {
  const { config, loaded, load, setAI } = useConfigStore();

  useEffect(() => {
    load();
  }, [load]);

  if (!loaded) {
    return (
      <div className="flex items-center justify-center py-20 text-[var(--color-text-muted)]">
        <Loader2 className="w-5 h-5 animate-spin mr-2" />
        Loading AI settings...
      </div>
    );
  }

  return (
    <AIProviderEditor
      settings={config.ai}
      onSave={async (settings) => {
        await setAI(settings);
        toast.success("AI provider saved");
      }}
    />
  );
}

interface AIProviderEditorProps {
  settings: AIProviderSettings;
  onSave: (settings: AIProviderSettings) => Promise<void>;
}

function AIProviderEditor({ settings, onSave }: AIProviderEditorProps) {
  const initialPreset = presetForBaseUrl(settings.baseUrl);

  const [providerKey, setProviderKey] = useState(initialPreset.key);
  const [baseUrl, setBaseUrl] = useState(settings.baseUrl || initialPreset.baseUrl);
  const [apiKey, setApiKey] = useState(settings.apiKey || "");
  const [model, setModel] = useState(settings.model || "");
  const [systemMessage, setSystemMessage] = useState(settings.systemMessage || "");
  const [transcriptionUrl, setTranscriptionUrl] = useState(settings.transcriptionBaseUrl || "");
  const [transcriptionModel, setTranscriptionModel] = useState(settings.transcriptionModel || "");
  const [transcriptionKey, setTranscriptionKey] = useState(settings.transcriptionApiKey || "");
  const [transcriptionDevice, setTranscriptionDevice] = useState<"auto" | "cpu" | "cuda">(
    settings.transcriptionDevice || "auto"
  );
  const [transcriptionModelPath, setTranscriptionModelPath] = useState(
    settings.transcriptionModelPath || ""
  );
  const [pathValidation, setPathValidation] = useState<LocalWhisperPathValidation | null>(null);
  const [validatingPath, setValidatingPath] = useState(false);
  const [localProbe, setLocalProbe] = useState<LocalWhisperProbe | null>(null);
  const [probingLocal, setProbingLocal] = useState(false);
  const [deletingModelDir, setDeletingModelDir] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>(() =>
    settings.model && !FALLBACK_MODELS.includes(settings.model)
      ? [settings.model, ...FALLBACK_MODELS]
      : FALLBACK_MODELS
  );
  const [loadingModels, setLoadingModels] = useState(false);
  const [saving, setSaving] = useState(false);

  const selectedPreset = useMemo(
    () => AI_PROVIDER_PRESETS.find((p) => p.key === providerKey) ?? AI_PROVIDER_PRESETS[0],
    [providerKey]
  );

  const isCustom = providerKey === "custom";
  const signupUrl = selectedPreset.signupUrl;

  // v2.0.87: transcription provider presets. Seed the picker from whatever is
  // already saved so an existing config keeps showing the right provider,
  // and fall back to Hugging Face rather than "custom" so the most common
  // setup is one click away.
  const [transcriptionPresetKey, setTranscriptionPresetKey] = useState(
    () => transcriptionPresetFor(settings.transcriptionBaseUrl || "", settings.transcriptionModel || "").key
  );
  const transcriptionPreset = useMemo(
    () => TRANSCRIPTION_PRESETS.find((p) => p.key === transcriptionPresetKey) ?? TRANSCRIPTION_PRESETS[0],
    [transcriptionPresetKey]
  );

  const refreshLocalProbe = useCallback(async () => {
    setProbingLocal(true);
    try {
      const info = await probeLocalWhisper();
      setLocalProbe(info);
    } catch (err) {
      setLocalProbe({
        available: false,
        version: null,
        reason: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setProbingLocal(false);
    }
  }, []);

  const validateManualPath = useCallback(async (rawPath: string) => {
    const trimmed = rawPath.trim();
    if (!trimmed) {
      setPathValidation(null);
      return;
    }
    setValidatingPath(true);
    try {
      const res = await validateLocalWhisperPath(trimmed);
      setPathValidation(res);
    } catch (err) {
      setPathValidation({
        valid: false,
        resolved_path: null,
        name: null,
        size_mb: 0,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setValidatingPath(false);
    }
  }, []);

  useEffect(() => {
    if (transcriptionPreset.local) {
      refreshLocalProbe();
      if (transcriptionModelPath.trim()) {
        validateManualPath(transcriptionModelPath);
      }
    }
  }, [transcriptionPreset.local, refreshLocalProbe, validateManualPath]);

  const handlePickManualModelFolder = async () => {
    const picked = await openDialog({
      directory: true,
      multiple: false,
      title: "Pilih Folder Model faster-whisper (CTranslate2)",
    });
    if (typeof picked === "string" && picked.trim()) {
      setTranscriptionModelPath(picked);
      await validateManualPath(picked);
    }
  };

  const handleDeleteCachedModel = async (dirName: string, displayName: string) => {
    setDeletingModelDir(dirName);
    try {
      const res = await deleteLocalWhisperModel(dirName);
      toast.success(`Cache ${displayName} dihapus (${res.freed_mb} MB dibebaskan)`);
      await refreshLocalProbe();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingModelDir(null);
    }
  };

  const handleTranscriptionPresetChange = (key: string) => {
    setTranscriptionPresetKey(key);
    const preset = TRANSCRIPTION_PRESETS.find((p) => p.key === key);
    if (!preset) return;
    // Prefill the fields, but NEVER touch the API key. It is provider-specific
    // and a typed credential is the user's property: silently discarding it
    // means re-pasting every time they toggle between local and cloud. The key
    // is simply ignored while local is selected (the field is hidden and the
    // sidecar never reads it on the local path), so keeping it is safe.
    setTranscriptionUrl(preset.baseUrl);
    setTranscriptionModel(preset.model);
  };

  const handleProviderChange = (value: string) => {
    const preset = AI_PROVIDER_PRESETS.find((p) => p.key === value) ?? AI_PROVIDER_PRESETS[0];
    setProviderKey(preset.key);
    setBaseUrl(preset.baseUrl);
  };

  const handleLoadModels = async () => {
    if (!apiKey.trim()) {
      toast.error("API key is required");
      return;
    }

    setLoadingModels(true);
    try {
      const loaded = await listAIModels(apiKey, baseUrl);
      setModels(loaded);
      if (loaded.length > 0 && !loaded.includes(model)) {
        setModel(loaded[0]);
      }
      toast.success(`Loaded ${loaded.length} models`);
    } catch (err) {
      console.error("Failed to load models", err);
      const detail = err instanceof Error ? err.message : String(err);
      toast.error("Failed to load models");
      setModels((prev) => (prev.length ? prev : FALLBACK_MODELS));
      alert(detail);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleSave = async () => {
    if (!apiKey.trim()) {
      toast.error("API key is required");
      return;
    }
    if (!model.trim()) {
      toast.error("Model is required");
      return;
    }

    setSaving(true);
    try {
      await onSave({
      baseUrl,
      apiKey,
      model,
      systemMessage,
      transcriptionBaseUrl: transcriptionUrl,
      transcriptionModel: transcriptionModel,
      transcriptionApiKey: transcriptionKey,
      transcriptionDevice,
      transcriptionModelPath: transcriptionModelPath.trim(),
    });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">
          AI Model
        </h1>
        <p className="text-sm text-[var(--color-text-muted)] mt-1">
          One AI provider for the whole app — used for both finding highlights and
          generating titles. Configure it once.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[var(--color-accent)]" />
            Provider Type
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <select
            value={providerKey}
            onChange={(e) => handleProviderChange(e.target.value)}
            className="w-full h-10 px-3 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-input)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-border-focus)]"
          >
            {AI_PROVIDER_PRESETS.map((provider) => (
              <option key={provider.key} value={provider.key}>
                {provider.name}
              </option>
            ))}
          </select>

          <p className="text-xs text-[var(--color-text-muted)]">
            {selectedPreset.description}
            {selectedPreset.apiKeyFormat
              ? ` · API key format: ${selectedPreset.apiKeyFormat}`
              : ""}
          </p>

          {signupUrl && (
            <div className="flex items-start gap-3 p-3 rounded-[var(--radius-sm)] bg-[var(--color-accent-light)] border border-[var(--color-accent)]/30">
              <Sparkles className="w-4 h-4 text-[var(--color-accent)] shrink-0 mt-0.5" />
              <div className="text-xs text-[var(--color-text-secondary)] space-y-1">
                <p>
                  Don't have an account yet? Get your API key at{" "}
                  <button
                    type="button"
                    onClick={() => openUrl(signupUrl)}
                    className="inline-flex items-center gap-1 text-[var(--color-accent)] font-medium hover:underline cursor-pointer"
                  >
                    {signupLabel(signupUrl)}
                    <ExternalLink className="w-3 h-3" />
                  </button>
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="w-5 h-5 text-[var(--color-accent)]" />
            API Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {isCustom ? (
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--color-text-secondary)]">
                Base URL
              </label>
              <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
            </div>
          ) : (
            <div className="space-y-1">
              <label className="text-sm font-medium text-[var(--color-text-secondary)]">
                Base URL
              </label>
              <div className="text-sm px-3 py-2 rounded-[var(--radius-sm)] bg-[var(--color-bg-secondary)] border border-[var(--color-border-light)] text-[var(--color-text-muted)]">
                {baseUrl}
              </div>
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text-secondary)]">
              API Key
            </label>
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={selectedPreset.apiKeyFormat ?? "Your API key"}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text-secondary)]">
              Model
            </label>
            <div className="flex gap-2">
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="flex-1 h-10 px-3 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-input)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-border-focus)]"
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <Button variant="outline" onClick={handleLoadModels} disabled={loadingModels}>
                {loadingModels ? <Loader2 className="w-4 h-4 animate-spin" /> : "Load"}
              </Button>
            </div>
            <Input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="Or type a model manually"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text-secondary)]">
              System Message
              <span className="ml-1 font-normal text-[var(--color-text-muted)]">
                (highlight finder only)
              </span>
            </label>
            <textarea
              value={systemMessage}
              onChange={(e) => setSystemMessage(e.target.value)}
              placeholder="Optional custom instructions for highlight detection"
              className="w-full min-h-[180px] px-3 py-2 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-bg-input)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-border-focus)] resize-y"
            />
            <p className="text-xs text-[var(--color-text-muted)]">
              Available placeholders: {"{num_clips}"}, {"{video_context}"},{" "}
              {"{transcript}"}, {"{user_direction}"}. Leave {"{user_direction}"} out and
              any direction typed on the Create page is appended at the end instead.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AudioLines className="w-5 h-5 text-[var(--color-accent)]" />
            Transcription (local videos)
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-[var(--color-text-muted)]">
            Dipakai HANYA saat transkripsi video lokal yang Bos upload ("File
            Lokal"). Provider AI utama belum tentu punya endpoint Whisper — jadi
            isi terpisah di sini. Biarkan kosong untuk memakai provider utama.
          </p>

          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text-secondary)]">
              Provider Transkripsi
            </label>
            <select
              value={transcriptionPresetKey}
              onChange={(e) => handleTranscriptionPresetChange(e.target.value)}
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg-secondary)] px-3 py-2 text-sm text-[var(--color-text-primary)]"
            >
              {TRANSCRIPTION_PRESETS.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.name}
                </option>
              ))}
            </select>
            <p className="text-xs text-[var(--color-text-muted)]">
              {transcriptionPreset.description} · API key:{" "}
              <code className="px-1 rounded bg-[var(--color-bg-secondary)]">
                {transcriptionPreset.apiKeyFormat}
              </code>
            </p>
            {transcriptionPreset.localNote ? (
              <p className="text-xs text-[var(--color-warning)] border-l-2 border-[var(--color-warning)] pl-2">
                {transcriptionPreset.localNote}
              </p>
            ) : null}
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text-secondary)]">
              Transcription Base URL
            </label>
            <Input
              value={transcriptionUrl}
              onChange={(e) => setTranscriptionUrl(e.target.value)}
              placeholder={transcriptionPreset.baseUrl}
              readOnly={transcriptionPreset.local}
              className={transcriptionPreset.local ? "opacity-60" : undefined}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-[var(--color-text-secondary)]">
              Transcription Model
            </label>
            {transcriptionPreset.models ? (
              <select
                value={transcriptionModel || transcriptionPreset.model}
                onChange={(e) => setTranscriptionModel(e.target.value)}
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg-secondary)] px-3 py-2 text-sm text-[var(--color-text-primary)]"
              >
                {transcriptionPreset.models.map((id) => {
                  const info = LOCAL_WHISPER_MODELS.find((m) => m.id === id);
                  return (
                    <option key={id} value={id}>
                      {id} · {(info?.sizeMB ?? 0).toLocaleString("id-ID")} MB —{" "}
                      {info?.note}
                    </option>
                  );
                })}
              </select>
            ) : (
              <Input
                value={transcriptionModel}
                onChange={(e) => setTranscriptionModel(e.target.value)}
                placeholder={transcriptionPreset.model}
              />
            )}
          </div>

          {transcriptionPreset.local ? (
            <div className="space-y-4">
              {/* Manual pre-downloaded local Whisper model folder */}
              <div className="space-y-2 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3">
                <label className="text-sm font-medium text-[var(--color-text-secondary)] flex items-center gap-1.5">
                  <FolderOpen className="w-4 h-4 text-[var(--color-accent)]" />
                  Folder Model Whisper Manual (Opsional — Tanpa Download)
                </label>
                <p className="text-xs text-[var(--color-text-muted)]">
                  Sudah punya model <strong>faster-whisper</strong> (format CTranslate2 yang berisi{" "}
                  <code className="px-1 rounded bg-[var(--color-bg-input)]">model.bin</code>) dari aplikasi
                  lain atau cache Hugging Face? Pilih foldernya di sini agar langsung dipakai secara offline.
                  Jika diisi, dropdown model di atas akan diabaikan.
                </p>
                <div className="flex gap-2">
                  <Input
                    value={transcriptionModelPath}
                    onChange={(e) => {
                      setTranscriptionModelPath(e.target.value);
                      if (!e.target.value.trim()) setPathValidation(null);
                    }}
                    onBlur={() => {
                      if (transcriptionModelPath.trim()) {
                        validateManualPath(transcriptionModelPath);
                      }
                    }}
                    placeholder="Contoh: D:\Models\faster-whisper-small atau ...\models--Systran--faster-whisper-turbo"
                    className="flex-1 text-xs"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handlePickManualModelFolder}
                    className="shrink-0 gap-1.5"
                  >
                    <FolderOpen className="w-4 h-4" />
                    Pilih Folder
                  </Button>
                  {transcriptionModelPath.trim() && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setTranscriptionModelPath("");
                        setPathValidation(null);
                      }}
                      title="Hapus path manual (kembali pakai dropdown)"
                      className="shrink-0 px-2"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  )}
                </div>

                {validatingPath ? (
                  <p className="text-xs text-[var(--color-text-muted)] flex items-center gap-1.5">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Memvalidasi folder model...
                  </p>
                ) : pathValidation ? (
                  pathValidation.valid ? (
                    <div className="text-xs text-emerald-600 dark:text-emerald-400 flex items-start gap-1.5">
                      <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium">
                          Model valid: {pathValidation.name} ({pathValidation.size_mb.toLocaleString("id-ID")} MB)
                        </p>
                        {pathValidation.resolved_path &&
                          pathValidation.resolved_path !== transcriptionModelPath.trim() && (
                            <p className="text-[11px] text-[var(--color-text-muted)] break-all">
                              Resolved: {pathValidation.resolved_path}
                            </p>
                          )}
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-[var(--color-error)] flex items-start gap-1.5">
                      <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                      <span>{pathValidation.error}</span>
                    </p>
                  )
                ) : null}
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-[var(--color-text-secondary)] flex items-center gap-1.5">
                  <Cpu className="w-4 h-4" />
                  Device Transkripsi Lokal
                </label>
                <select
                  value={transcriptionDevice}
                  onChange={(e) =>
                    setTranscriptionDevice(e.target.value as "auto" | "cpu" | "cuda")
                  }
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg-secondary)] px-3 py-2 text-sm text-[var(--color-text-primary)]"
                >
                  <option value="auto">Auto (GPU CUDA 12 jika siap, fallback ke CPU/int8)</option>
                  <option value="cpu">CPU (int8 — paling stabil di semua PC)</option>
                  <option value="cuda">CUDA (float16 — wajib CUDA Toolkit 12 terpasang)</option>
                </select>
              </div>

              <div className="rounded-md border border-[var(--color-border)] bg-[var(--color-bg-secondary)] p-3 space-y-2.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-xs font-semibold text-[var(--color-text-primary)]">
                    <HardDrive className="w-4 h-4 text-[var(--color-accent)]" />
                    Status Engine & Cache Model Lokal
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={refreshLocalProbe}
                    disabled={probingLocal}
                    className="h-7 px-2 text-xs gap-1"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${probingLocal ? "animate-spin" : ""}`} />
                    Cek Ulang
                  </Button>
                </div>

                {probingLocal && !localProbe ? (
                  <p className="text-xs text-[var(--color-text-muted)]">
                    Memeriksa status faster-whisper...
                  </p>
                ) : localProbe ? (
                  <div className="space-y-2 text-xs">
                    {localProbe.available ? (
                      <div className="space-y-1">
                        <p className="text-[var(--color-text-primary)]">
                          Status:{" "}
                          <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                            Siap (faster-whisper v{localProbe.version})
                          </span>{" "}
                          · Aktif:{" "}
                          <code className="px-1 rounded bg-[var(--color-bg-input)]">
                            {localProbe.device}/{localProbe.compute_type}
                          </code>
                        </p>
                        {!localProbe.cuda_ready && localProbe.cuda_reason ? (
                          <p className="text-[var(--color-text-muted)]">
                            Info CUDA: {localProbe.cuda_reason} — mode Auto otomatis memakai CPU.
                          </p>
                        ) : null}
                      </div>
                    ) : (
                      <p className="text-[var(--color-danger)]">
                        Engine lokal belum siap: {localProbe.reason}
                      </p>
                    )}

                    <div className="pt-1 border-t border-[var(--color-border)] space-y-1.5">
                      <p className="font-medium text-[var(--color-text-secondary)]">
                        Model Terunduh ({localProbe.cached_models?.length ?? 0}) · Total{" "}
                        {(localProbe.cache_total_mb ?? 0).toLocaleString("id-ID")} MB
                      </p>
                      {localProbe.cached_models && localProbe.cached_models.length > 0 ? (
                        <div className="space-y-1">
                          {localProbe.cached_models.map((m) => (
                            <div
                              key={m.dir_name}
                              className="flex items-center justify-between gap-2 px-2 py-1.5 rounded bg-[var(--color-bg-input)]"
                            >
                              <div className="min-w-0">
                                <p className="truncate font-medium text-[var(--color-text-primary)]">
                                  {m.name}
                                </p>
                                <p className="text-[11px] text-[var(--color-text-muted)]">
                                  {m.size_mb.toLocaleString("id-ID")} MB
                                </p>
                              </div>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => handleDeleteCachedModel(m.dir_name, m.name)}
                                disabled={deletingModelDir === m.dir_name}
                                className="h-7 px-2 text-xs gap-1 shrink-0"
                              >
                                {deletingModelDir === m.dir_name ? (
                                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                ) : (
                                  <Trash2 className="w-3.5 h-3.5" />
                                )}
                                Hapus
                              </Button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[var(--color-text-muted)]">
                          Belum ada model yang diunduh. Model akan otomatis diunduh saat transkripsi pertama.
                        </p>
                      )}
                    </div>
                  </div>
                ) : null}
              </div>

              <p className="text-xs text-[var(--color-text-muted)]">
                Tidak perlu API key — mesinnya jalan di dalam aplikasi ini.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <label className="text-sm font-medium text-[var(--color-text-secondary)]">
                Transcription API Key
              </label>
              <Input
                type="password"
                value={transcriptionKey}
                onChange={(e) => setTranscriptionKey(e.target.value)}
                placeholder="Kosongkan untuk memakai API key provider utama"
              />
              <a
                href={transcriptionPreset.signupUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-block text-xs underline text-[var(--color-accent)]"
              >
                Ambil API key di {signupLabel(transcriptionPreset.signupUrl)} →
              </a>
            </div>
          )}
        </CardContent>
      </Card>

      <Button onClick={handleSave} disabled={saving} className="w-full h-12 gap-2">
        {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
        Save AI Provider
      </Button>
    </div>
  );
}
