import type { AnalysisResult, Grain, GrainStatus } from "./types";

const API = "";

export async function analyzeFile(
  file: File,
  mode: "panorama" | "detail" | "auto"
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);
  if (mode !== "auto") {
    form.append("mode", mode);
  }

  const res = await fetch(`${API}/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Ошибка анализа");
  }
  return res.json();
}

export async function applyCorrections(
  resultId: string,
  grains: { id: number; status?: GrainStatus; bbox?: number[] }[]
): Promise<AnalysisResult> {
  const res = await fetch(`${API}/result/${resultId}/corrections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ grains }),
  });
  if (!res.ok) {
    throw new Error("Не удалось сохранить правки");
  }
  const data = await res.json();
  return { ...data, result_id: data.result_id };
}

export async function applyTalcMask(resultId: string, maskPng: Blob): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("mask", maskPng, "mask.png");
  const res = await fetch(`${API}/result/${resultId}/talc-mask`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error("Не удалось сохранить маску талька");
  }
  const data = await res.json();
  return { ...data, result_id: data.result_id };
}

export function absUrl(path: string | null): string {
  if (!path) return "";
  return path.startsWith("http") ? path : `${API}${path}`;
}

export function statusLabel(s: GrainStatus): string {
  const map: Record<GrainStatus, string> = {
    ordinary: "рядовое",
    thin: "тонкое",
    uncertain: "неопределённый",
    false_positive: "ложная детекция",
  };
  return map[s] || s;
}

export function grainColor(g: Grain): string {
  const s = g.status || g.intergrowth_type;
  if (s === "false_positive") return "transparent";
  if (s === "thin") return "rgba(220, 40, 40, 0.7)";
  if (s === "uncertain") return "rgba(200, 200, 0, 0.7)";
  return "rgba(0, 200, 0, 0.7)";
}
