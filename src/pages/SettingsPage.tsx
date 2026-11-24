import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Cpu, HardDrive, Loader2, RefreshCw, Trash2, Zap, ChevronRight, Sun, Moon } from "lucide-react";
import { useAppStore } from "@/stores/appStore";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { SettingRow, SettingSection } from "@/components/settings/SettingRow";
import {
  cleanupStorage,
  detectGpu,
  getStorageSummary,
  type GpuDetection,
  type StorageSummary,
} from "@/hooks/appConfig";
import { useConfigStore } from "@/stores/configStore";

export function SettingsPage() {
  const navigate = useNavigate();
  const { config, loaded, load, setGpuAcceleration } = useConfigStore();
  const { theme, setTheme, showLogs, toggleShowLogs } = useAppStore();
  const [detection, setDetection] = useState<GpuDetection | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [storage, setStorage] = useState<StorageSummary | null>(null);
  const [loadingStorage, setLoadingStorage] = useState(false);
  const [cleaningStorage, setCleaningStorage] = useState(false);

  const loadStorage = async () => {
    setLoadingStorage(true);
    try {
      const summary = await getStorageSummary();
      setStorage(summary);
    } catch (err) {
      console.error("Storage summary failed", err);
    } finally {
      setLoadingStorage(false);
    }
  };

  useEffect(() => {
    load();
  }, [load]);

  const runDetect = async () => {
    setDetecting(true);
    try {
      const result = await detectGpu();
      setDetection(result);
    } catch (err) {
      console.error("GPU detection failed", err);
      toast.error("GPU detection failed");
    } finally {
      setDetecting(false);
    }
  };

  useEffect(() => {
    runDetect();
    loadStorage();
  }, []);

  const gpuEnabled = config.gpuAcceleration.enabled;
  const encoderAvailable = detection?.encoder.available ?? false;
  const canToggle = encoderAvailable && !detecting;

  const handleToggle = async (next: boolean) => {
    setSaving(true);
    try {
      // Persist the detected encoder + preset alongside enabled — the sidecar
      // needs them to build ffmpeg args. Saving only { enabled: true } made the
      // backend fall back to libx264 (CPU) even though the toggle was on.
      await setGpuAcceleration({
        enabled: next,
        encoder: next ? detection?.encoder.name ?? null : null,
        preset: next ? detection?.encoder.preset ?? null : null,
      });
      toast.success(next ? "GPU acceleration enabled" : "GPU acceleration disabled");
    } catch {
      toast.error("Failed to save setting");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Settings</h1>
        <p className="text-sm text-[var(--color-text-muted)] mt-1">
          Configure performance and output options.
        </p>
      </div>

      <SettingSection
        title="Appearance"
        description="Theme and display preferences."
      >
        <SettingRow
          title="Theme"
          description={
            theme === "dark"
              ? "Dark mode (pekat) — easier on the eyes at night"
              : "Light mode — bright and clean"
          }
        >
          <div className="flex items-center gap-1.5">
            <Button
              variant={theme === "light" ? "default" : "outline"}
              size="sm"
              onClick={() => setTheme("light")}
              className="gap-1.5"
            >
              <Sun className="w-4 h-4" />
              Light
            </Button>
            <Button
              variant={theme === "dark" ? "default" : "outline"}
              size="sm"
              onClick={() => setTheme("dark")}
              className="gap-1.5"
            >
              <Moon className="w-4 h-4" />
              Dark
            </Button>
          </div>
        </SettingRow>

        <SettingRow
          title="Log output"
          description={
            showLogs
              ? "Show the debug log console during processing"
              : "Hide the debug log console during processing"
          }
        >
          <Switch checked={showLogs} onCheckedChange={toggleShowLogs} />
        </SettingRow>
      </SettingSection>

      <SettingSection
        title="GPU Acceleration"
        description="Hardware encoding is 3-5x faster than CPU. Requires a compatible GPU."
      >
        <SettingRow
          title="Detected GPU"
          description={
            detecting
              ? "Detecting..."
              : detection
                ? detection.gpu.available
                  ? `${gpuTypeLabel(detection.gpu.type)} · ${detection.gpu.name}`
                  : "No GPU detected — CPU encoding will be used"
                : "Not checked yet"
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={runDetect}
            disabled={detecting}
            className="gap-2"
          >
            {detecting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            Detect
          </Button>
        </SettingRow>

        <SettingRow
          title="Enable GPU acceleration"
          description={
            encoderAvailable
              ? `Encoder: ${detection?.encoder.name}${
                  detection?.encoder.preset ? ` · preset ${detection.encoder.preset}` : ""
                }`
              : detection?.encoder.reason ?? "Run detection to check encoder support"
          }
        >
          <div className="flex items-center gap-2">
            {saving && <Loader2 className="w-4 h-4 animate-spin text-[var(--color-text-muted)]" />}
            <Switch
              checked={gpuEnabled && encoderAvailable}
              onCheckedChange={handleToggle}
              disabled={!canToggle || saving || !loaded}
            />
          </div>
        </SettingRow>

        <SettingRow
          title="Active encoder"
          description="What will be used when processing video."
        >
          <span className="inline-flex items-center gap-2 text-sm font-medium">
            {gpuEnabled && encoderAvailable ? (
              <>
                <Zap className="w-4 h-4 text-[var(--color-success)]" />
                <span className="text-[var(--color-success)]">
                  {detection?.encoder.name}
                </span>
              </>
            ) : (
              <>
                <Cpu className="w-4 h-4 text-[var(--color-text-muted)]" />
                <span className="text-[var(--color-text-secondary)]">libx264 (CPU)</span>
              </>
            )}
          </span>
        </SettingRow>
      </SettingSection>

      <SettingSection
        title="Storage Manager"
        description="Pantau dan bersihkan cache video sumber (Download Once), model Whisper lokal, dan session."
      >
        <SettingRow
          title="Total Pemakaian Disk"
          description={
            loadingStorage && !storage
              ? "Menghitung ukuran cache..."
              : storage
                ? `Total ${storage.total_mb.toLocaleString("id-ID")} MB · Source Videos: ${storage.source_videos.size_mb.toLocaleString("id-ID")} MB (${storage.source_videos.count} file) · Sessions: ${storage.sessions.size_mb.toLocaleString("id-ID")} MB (${storage.sessions.count}) · Whisper Models: ${storage.whisper_models.size_mb.toLocaleString("id-ID")} MB (${storage.whisper_models.count})`
                : "Klik Refresh untuk memeriksa ukuran cache."
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={loadStorage}
            disabled={loadingStorage}
            className="gap-1.5"
          >
            <RefreshCw className={`w-4 h-4 ${loadingStorage ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </SettingRow>

        <SettingRow
          title="Bersihkan Source Video Lama (> 7 hari)"
          description="Menghapus cache full video YouTube yang lebih tua dari 7 hari. Klip yang sudah dirender tetap aman."
        >
          <Button
            variant="outline"
            size="sm"
            disabled={cleaningStorage || !storage || storage.source_videos.count === 0}
            onClick={async () => {
              setCleaningStorage(true);
              try {
                const res = await cleanupStorage({
                  action: "source_videos_older_than",
                  olderThanDays: 7,
                });
                toast.success(
                  res.deleted_count > 0
                    ? `${res.deleted_count} source video dihapus (${res.freed_mb} MB dibebaskan)`
                    : "Tidak ada source video yang lebih tua dari 7 hari"
                );
                await loadStorage();
              } catch (err) {
                toast.error(err instanceof Error ? err.message : String(err));
              } finally {
                setCleaningStorage(false);
              }
            }}
            className="gap-1.5"
          >
            {cleaningStorage ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
            Bersihkan &gt; 7 Hari
          </Button>
        </SettingRow>

        {storage && storage.source_videos.items.length > 0 && (
          <div className="px-4 py-3 border-t border-[var(--color-border-light)] space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium text-[var(--color-text-secondary)] flex items-center gap-1.5">
                <HardDrive className="w-3.5 h-3.5" />
                Cached Source Videos ({storage.source_videos.items.length})
              </p>
              <Button
                variant="ghost"
                size="sm"
                disabled={cleaningStorage}
                onClick={async () => {
                  if (!window.confirm("Hapus semua cache source video? Klip yang sudah dirender tetap aman.")) return;
                  setCleaningStorage(true);
                  try {
                    const res = await cleanupStorage({ action: "all_source_videos" });
                    toast.success(`${res.deleted_count} file dihapus (${res.freed_mb} MB dibebaskan)`);
                    await loadStorage();
                  } catch (err) {
                    toast.error(err instanceof Error ? err.message : String(err));
                  } finally {
                    setCleaningStorage(false);
                  }
                }}
                className="h-7 px-2 text-xs text-[var(--color-error)]"
              >
                Hapus Semua
              </Button>
            </div>
            <div className="max-h-40 overflow-y-auto space-y-1">
              {storage.source_videos.items.map((item) => (
                <div
                  key={item.filename}
                  className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded bg-[var(--color-bg-secondary)] text-xs"
                >
                  <div className="min-w-0">
                    <p className="truncate font-medium text-[var(--color-text-primary)]">{item.filename}</p>
                    <p className="text-[11px] text-[var(--color-text-muted)]">
                      {item.size_mb.toLocaleString("id-ID")} MB · {item.age_days} hari lalu
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={cleaningStorage}
                    onClick={async () => {
                      setCleaningStorage(true);
                      try {
                        const res = await cleanupStorage({
                          action: "source_video_file",
                          filename: item.filename,
                        });
                        toast.success(`${item.filename} dihapus (${res.freed_mb} MB)`);
                        await loadStorage();
                      } catch (err) {
                        toast.error(err instanceof Error ? err.message : String(err));
                      } finally {
                        setCleaningStorage(false);
                      }
                    }}
                    className="h-7 px-2 text-xs shrink-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}
      </SettingSection>

      <SettingSection
        title="Output"
        description="Branding and overlays applied to every clip."
      >
        <SettingRow
          title="Watermark"
          description={
            config.watermark.enabled
              ? "Enabled — overlay a logo on each clip"
              : "Add a logo overlay to each clip"
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/settings/watermark")}
            className="gap-2"
          >
            Configure
            <ChevronRight className="w-4 h-4" />
          </Button>
        </SettingRow>

        <SettingRow
          title="Credit Text"
          description={
            config.creditWatermark.enabled
              ? "Enabled — credit overlay on each clip"
              : "Add source channel credit overlay"
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/settings/credit")}
            className="gap-2"
          >
            Configure
            <ChevronRight className="w-4 h-4" />
          </Button>
        </SettingRow>

        <SettingRow
          title="Hook Style"
          description={
            "Font, colors, and shape of the opening hook scene"
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/settings/hook-style")}
            className="gap-2"
          >
            Configure
            <ChevronRight className="w-4 h-4" />
          </Button>
        </SettingRow>

        <SettingRow
          title="Intro & Outro"
          description={
            config.introOutro.enabled
              ? "Enabled — smooth transitions between clips"
              : "Prepend an intro and append an outro to clips"
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/settings/intro-outro")}
            className="gap-2"
          >
            Configure
            <ChevronRight className="w-4 h-4" />
          </Button>
        </SettingRow>
      </SettingSection>

      <SettingSection
        title="Integrations"
        description="Connect external services for publishing your clips."
      >
        <SettingRow
          title="Repliz"
          description={
            config.repliz.accessKey && config.repliz.secretKey
              ? "Connected — upload clips to multiple platforms"
              : "Upload clips to YouTube, TikTok, Instagram, and more"
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/settings/repliz")}
            className="gap-2"
          >
            Configure
            <ChevronRight className="w-4 h-4" />
          </Button>
        </SettingRow>
      </SettingSection>

      <SettingSection
        title="About"
        description="Application information and credits."
      >
        <SettingRow
          title="Open Source Credits"
          description="Libraries and tools used in this application"
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/settings/credits")}
            className="gap-2"
          >
            View
            <ChevronRight className="w-4 h-4" />
          </Button>
        </SettingRow>
      </SettingSection>
    </div>
  );
}

function gpuTypeLabel(type: string | null): string {
  switch (type) {
    case "nvidia":
      return "NVIDIA";
    case "amd":
      return "AMD";
    case "intel":
      return "Intel";
    case "apple":
      return "Apple";
    default:
      return "GPU";
  }
}
