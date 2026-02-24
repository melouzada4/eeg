# ============================================================
# EEG ANALYSIS (HÍBRIDO "ESTILO ANTIGO" + METODOLOGIA ROBUSTA)
# - PSD por intervalo -> média do PSD por condição/canal -> FOOOF 1x
# - Gráficos zoomados por banda + caixa de estatísticas
# - Opção A: ID na planilha = nome do arquivo .set (sem extensão)
# - Log JSON + TXT resumo por canal/condição
# ============================================================

import os
import json
import platform
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from fooof import FOOOF

# =========================
# SILENCIAR AVISOS
# =========================
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^fooof(\.|$)")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*fooof.*deprecated.*")

# =========================
# CONFIGURAÇÕES
# =========================
DURACAO_INTERVALO = 6  # segundos
NUM_INTERVALOS_OA = 10
NUM_INTERVALOS_OF = 10

# Range global do FOOOF (coerente com teu filtro final)
FMIN_GLOBAL = 1.0
FMAX_GLOBAL = 40.0

# Bandas para métricas/plots (zoom)
BANDAS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Welch (estável)
WELCH_WIN_SEC = 2.0
WELCH_OVERLAP_FRAC = 0.5
NFFT_MIN = 2048

# FOOOF (ajuste global)
FOOOF_CONFIG = {
    "peak_width_limits": [1, 12],
    "max_n_peaks": 3,
    "min_peak_height": 0.01,
    "peak_threshold": 2.0,
    "aperiodic_mode": "fixed",
    "verbose": False,
}

# Quality gate FOOOF (aplicado ao PSD médio)
FOOOF_R2_MIN = 0.70
FOOOF_ERR_MAX = 0.30

# =========================
# *** CAMINHOS ALTERADOS ***
# (no Código 2 original havia GUI via tkinter para selecionar
#  estes três valores interativamente; aqui foram substituídos
#  por caminhos fixos do Código 1)
# =========================
PASTA_SET     = r'C:/Users/meloliveira/Documents/eeg'                    
PLANILHA_PATH = r'C:/Users/meloliveira/Documents/intervalos_pacientes.xlsx'    
SUBPASTA_SAIDA = "sol2"
PASTA_SAIDA   = os.path.join(PASTA_SET, SUBPASTA_SAIDA)                    
os.makedirs(PASTA_SAIDA, exist_ok=True)                                        


# =========================
# PLANILHA (OPÇÃO A)
# =========================
def carregar_intervalos_planilha(caminho_planilha):
    """
    Formato (como tua imagem):
    Coluna A: ID (ex.: 42_T0FilterData)
    Colunas B..K: trecho 1..10 (OA)
    Colunas L..U: trecho 1..10 (OF)
    """
    df = pd.read_excel(caminho_planilha)
    col_id = df.columns[0]

    intervalos = {}
    for _, row in df.iterrows():
        id_str = str(row[col_id]).strip()
        if not id_str or id_str.lower() == "nan":
            continue

        oa = []
        for i in range(1, 11):  # B..K
            if i < len(row) and not pd.isna(row.iloc[i]):
                oa.append(int(row.iloc[i]))

        of = []
        for i in range(11, 21):  # L..U
            if i < len(row) and not pd.isna(row.iloc[i]):
                of.append(int(row.iloc[i]))

        if len(oa) >= 10 and len(of) >= 10:
            intervalos[id_str] = {"OA": oa[:10], "OF": of[:10]}

    return intervalos


# =========================
# UTIL
# =========================
def salvar_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def extrair_segmento(sig_1d, sfreq, inicio_s, dur_s):
    ini = int(round(inicio_s * sfreq))
    fim = int(round((inicio_s + dur_s) * sfreq))
    ini = max(0, ini)
    fim = min(sig_1d.shape[0], fim)
    if fim <= ini:
        return None
    return sig_1d[ini:fim]


def welch_psd(sig_1d, sfreq):
    """
    Retorna freqs, psd (V²/Hz) em linear.
    """
    n = sig_1d.shape[0]
    if n < int(sfreq):  # <1s
        return None, None

    nperseg = int(round(WELCH_WIN_SEC * sfreq))
    nperseg = min(nperseg, n)
    nperseg = max(int(sfreq), nperseg)  # >=1s

    noverlap = int(round(WELCH_OVERLAP_FRAC * nperseg))
    noverlap = min(noverlap, nperseg - 1)

    nfft = max(int(4 * sfreq), NFFT_MIN)

    freqs, pxx = welch(sig_1d, fs=sfreq, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return freqs, pxx


def idx_range(freqs, fmin, fmax):
    return (freqs >= fmin) & (freqs <= fmax)


# =========================
# FOOOF NO PSD MÉDIO
# =========================
def ajustar_fooof_no_psd_medio(freqs, psd_med, fmin=FMIN_GLOBAL, fmax=FMAX_GLOBAL):
    """
    Ajusta FOOOF uma vez no PSD médio (V²/Hz).
    Retorna dict com curvas (µV²/Hz) e métricas.
    """
    idx = idx_range(freqs, fmin, fmax)
    if idx.sum() < 20:
        return None

    f = freqs[idx]
    p = psd_med[idx]

    if np.any(~np.isfinite(p)) or np.any(p <= 0):
        return None

    fm = FOOOF(**FOOOF_CONFIG)
    fm.fit(f, p)

    r2 = float(getattr(fm, "r_squared_", 0.0) or 0.0)
    err = float(getattr(fm, "error_", np.inf) or np.inf)

    ok = (r2 >= FOOOF_R2_MIN) and (err <= FOOOF_ERR_MAX)

    orig_micro = p * 1e12
    ap_micro = (10 ** fm._ap_fit) * 1e12
    full_micro = (10 ** fm.fooofed_spectrum_) * 1e12
    periodic_micro = full_micro - ap_micro

    return {
        "ok": ok,
        "fm": fm,
        "freqs": f,
        "orig_micro": orig_micro,
        "ap_micro": ap_micro,
        "full_micro": full_micro,
        "periodic_micro": periodic_micro,
        "r2": r2,
        "err": err,
    }


def metricas_por_banda(freqs, orig_micro, ap_micro, periodic_micro, fmin, fmax):
    idx = idx_range(freqs, fmin, fmax)
    if idx.sum() < 5:
        return None

    f = freqs[idx]
    o = orig_micro[idx]
    a = ap_micro[idx]
    p = periodic_micro[idx]

    return {
        "psd_media_original": float(np.mean(o)),
        "area_original": float(np.trapz(o, f)),
        "psd_media_aperiodic": float(np.mean(a)),
        "area_aperiodic": float(np.trapz(a, f)),
        "psd_media_periodic": float(np.mean(p)),
        "area_periodic": float(np.trapz(p, f)),
    }


def resumo_picos_faixa(fm, fmin, fmax):
    peaks = getattr(fm, "peak_params_", None)
    if peaks is None or len(peaks) == 0:
        return {"n": 0, "freq_media": None, "amp_media": None, "bw_media": None}

    peaks = np.array(peaks)
    freqs = peaks[:, 0]
    amps = peaks[:, 1]
    bws = peaks[:, 2]

    m = (freqs >= fmin) & (freqs <= fmax)
    if m.sum() == 0:
        return {"n": 0, "freq_media": None, "amp_media": None, "bw_media": None}

    return {
        "n": int(m.sum()),
        "freq_media": float(np.mean(freqs[m])),
        "amp_media": float(np.mean(amps[m])),
        "bw_media": float(np.mean(bws[m])),
    }


# =========================
# PLOT (ZOOM + CAIXA STATS)
# =========================
def plot_zoom_banda_com_stats(resultado_fooof, banda_nome, canal, cond, paciente_id,
                             pasta_saida, fmin_b, fmax_b, n_validos, n_total=10):
    f = resultado_fooof["freqs"]
    orig = resultado_fooof["orig_micro"]
    ap = resultado_fooof["ap_micro"]
    full = resultado_fooof["full_micro"]
    per = resultado_fooof["periodic_micro"]
    fm = resultado_fooof["fm"]
    r2 = resultado_fooof["r2"]
    err = resultado_fooof["err"]

    idx = idx_range(f, fmin_b, fmax_b)
    if idx.sum() < 5:
        return

    fz = f[idx]
    origz = orig[idx]
    apz = ap[idx]
    fullz = full[idx]
    perz = per[idx]

    met = metricas_por_banda(f, orig, ap, per, fmin_b, fmax_b) or {}
    peaks = resumo_picos_faixa(fm, fmin_b, fmax_b)

    if len(perz) > 0:
        i_pk = int(np.argmax(perz))
        pk_freq_vis = float(fz[i_pk])
        pk_val_vis = float(perz[i_pk])
    else:
        pk_freq_vis, pk_val_vis = None, None

    titulo_cond = "Olhos Abertos" if cond == "OA" else "Olhos Fechados"

    plt.figure(figsize=(12, 7))
    plt.style.use("default")

    plt.plot(fz, origz, linewidth=1.3, label="Original")
    plt.plot(fz, fullz, linewidth=2.0, label="FOOOF full fit")
    plt.plot(fz, apz, linewidth=2.0, linestyle="--", label="Aperiódico")
    plt.fill_between(fz, apz, fullz, alpha=0.25, label="Periódico")

    plt.title(
        f"Banda {banda_nome.upper()} - Canal {canal} - {titulo_cond} - {paciente_id} "
        f"({n_validos}/{n_total})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Hz")
    plt.ylabel("Densidade Espectral de Potência (µV²/Hz)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right")

    stats_lines = []
    stats_lines.append(f"Estatísticas {banda_nome.upper()} ({fmin_b:.1f}-{fmax_b:.1f} Hz)")
    stats_lines.append("")
    if met:
        stats_lines.append("ORIGINAL:")
        stats_lines.append(f"  PSD média: {met.get('psd_media_original', float('nan')):.4f} µV²/Hz")
        stats_lines.append(f"  Área:      {met.get('area_original', float('nan')):.4f} µV²")
        stats_lines.append("")
        stats_lines.append("APERIÓDICO:")
        stats_lines.append(f"  PSD média: {met.get('psd_media_aperiodic', float('nan')):.4f} µV²/Hz")
        stats_lines.append(f"  Área:      {met.get('area_aperiodic', float('nan')):.4f} µV²")
        stats_lines.append("")
        stats_lines.append("PERIÓDICO:")
        stats_lines.append(f"  PSD média: {met.get('psd_media_periodic', float('nan')):.4f} µV²/Hz")
        stats_lines.append(f"  Área:      {met.get('area_periodic', float('nan')):.4f} µV²")
        stats_lines.append("")
    stats_lines.append("FOOOF:")
    stats_lines.append(f"  OK (gate): {resultado_fooof.get('ok', False)}")
    stats_lines.append(f"  R²:   {r2:.4f}")
    stats_lines.append(f"  Erro: {err:.4f}")
    stats_lines.append(f"  Picos na banda: {peaks['n']}")
    if peaks["n"] > 0 and peaks["freq_media"] is not None:
        stats_lines.append(f"  Freq média picos: {peaks['freq_media']:.2f} Hz")

    if pk_freq_vis is not None:
        stats_lines.append("")
        stats_lines.append("Pico periódico (visual):")
        stats_lines.append(f"  Freq: {pk_freq_vis:.2f} Hz")
        stats_lines.append(f"  PSD:  {pk_val_vis:.4f} µV²/Hz")

    box_text = "\n".join(stats_lines)

    ax = plt.gca()
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9)
    plt.text(
        0.98, 0.98, box_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=bbox_props
    )

    plt.tight_layout()
    nome_png = f"{paciente_id}_{canal}_{banda_nome}_{cond}_FOOOF_MEDIA.png"
    plt.savefig(os.path.join(pasta_saida, nome_png), dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# TXT RESUMO
# =========================
def salvar_txt_resumo(pasta_saida, paciente_id, canal, cond, n_validos, resultados_por_banda, fooof_info):
    caminho = os.path.join(pasta_saida, f"{paciente_id}_{canal}_{cond}_RESUMO.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(f"RESUMO EEG/FOOOF - {paciente_id}\n")
        f.write(f"Canal: {canal}\n")
        f.write(f"Condição: {cond}\n")
        f.write(f"Intervalos válidos: {n_validos}/10\n")
        f.write("-" * 70 + "\n")
        f.write("FOOOF no PSD MÉDIO (1–40 Hz):\n")
        f.write(f"  OK (quality gate): {fooof_info['ok']}\n")
        f.write(f"  R²: {fooof_info['r2']:.4f}\n")
        f.write(f"  Erro: {fooof_info['err']:.4f}\n\n")

        for banda, met in resultados_por_banda.items():
            fmin_b, fmax_b = BANDAS[banda]
            f.write(f"BANDA {banda.upper()} ({fmin_b}-{fmax_b} Hz)\n")
            if met is None:
                f.write("  Sem métricas (dados insuficientes)\n\n")
                continue
            f.write(f"  ORIGINAL:  PSD média {met['psd_media_original']:.5f} µV²/Hz | Área {met['area_original']:.5f} µV²\n")
            f.write(f"  APERIOD.:  PSD média {met['psd_media_aperiodic']:.5f} µV²/Hz | Área {met['area_aperiodic']:.5f} µV²\n")
            f.write(f"  PERIOD.:   PSD média {met['psd_media_periodic']:.5f} µV²/Hz | Área {met['area_periodic']:.5f} µV²\n\n")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # --- Carregar planilha (antes era retorno da GUI) ---
    intervalos_planilha = carregar_intervalos_planilha(PLANILHA_PATH) 
    if not intervalos_planilha:
        raise SystemExit("❌ Não foi possível ler intervalos da planilha.")

    log_geral = {
        "timestamp": datetime.now().isoformat(),
        "planilha": PLANILHA_PATH,                                      
        "saida": PASTA_SAIDA,                                         
        "parametros": {
            "duracao_intervalo_s": DURACAO_INTERVALO,
            "fmin_global": FMIN_GLOBAL,
            "fmax_global": FMAX_GLOBAL,
            "bandas": BANDAS,
            "welch_win_sec": WELCH_WIN_SEC,
            "welch_overlap_frac": WELCH_OVERLAP_FRAC,
            "nfft_min": NFFT_MIN,
            "fooof_config": FOOOF_CONFIG,
            "fooof_r2_min": FOOOF_R2_MIN,
            "fooof_err_max": FOOOF_ERR_MAX,
        },
        "ambiente": {
            "python": platform.python_version(),
            "mne": mne.__version__,
            "platform": platform.platform(),
        },
        "arquivos_processados": []
    }

    print("\n" + "=" * 80)
    print("INICIANDO ANÁLISE (PSD MÉDIO -> FOOOF 1x | GRÁFICO ZOOM + STATS)")
    print("=" * 80)

    # --- Loop pelos .set na pasta (antes iterava sobre lista da GUI) ---
    for nome_arquivo in os.listdir(PASTA_SET):                           # <- ALTERADO
        if not nome_arquivo.endswith(".set"):
            continue

        caminho = os.path.join(PASTA_SET, nome_arquivo)                  # <- ALTERADO
        paciente_id = Path(caminho).stem

        # Busca parcial: como no Código 1
        chave_encontrada = None
        if paciente_id in intervalos_planilha:
            chave_encontrada = paciente_id
        else:
            for chave in intervalos_planilha:
                if str(chave) in paciente_id or paciente_id in str(chave):
                    chave_encontrada = chave
                    break

        if chave_encontrada is None:
            print(f"⚠️ ID '{paciente_id}' não encontrado na planilha — pulando {nome_arquivo}")
            continue

        print(f"\n▶ Processando: {nome_arquivo} | ID: {paciente_id}")

        item_log = {
            "id": paciente_id,
            "arquivo": caminho,
            "processado": True,
            "sfreq": None,
            "duracao_s": None,
            "n_canais": None,
            "falhas_intervalo_psd": 0,
            "canais": {}
        }

        try:
            raw = mne.io.read_raw_eeglab(caminho, preload=True)
            data = raw.get_data()
            sfreq = float(raw.info["sfreq"])
            ch_names = list(raw.info["ch_names"])
            dur_total = data.shape[1] / sfreq

            item_log["sfreq"] = sfreq
            item_log["duracao_s"] = float(dur_total)
            item_log["n_canais"] = int(len(ch_names))

            ints = intervalos_planilha[chave_encontrada]  # {"OA":[...], "OF":[...]}

            for ch_idx, canal in enumerate(ch_names):
                sig = data[ch_idx, :]
                if np.all(sig == 0):
                    continue

                item_log["canais"][canal] = {}

                for cond in ["OA", "OF"]:
                    inicios = ints[cond]
                    psds = []
                    freqs_ref = None
                    usados = 0

                    # PSD por intervalo
                    for inicio_s in inicios:
                        if inicio_s + DURACAO_INTERVALO > dur_total:
                            item_log["falhas_intervalo_psd"] += 1
                            continue

                        seg = extrair_segmento(sig, sfreq, inicio_s, DURACAO_INTERVALO)
                        if seg is None or seg.size < int(sfreq):
                            item_log["falhas_intervalo_psd"] += 1
                            continue

                        freqs, pxx = welch_psd(seg, sfreq)
                        if freqs is None or pxx is None:
                            item_log["falhas_intervalo_psd"] += 1
                            continue

                        idxg = idx_range(freqs, FMIN_GLOBAL, FMAX_GLOBAL)
                        if idxg.sum() < 20:
                            item_log["falhas_intervalo_psd"] += 1
                            continue

                        freqs_g = freqs[idxg]
                        pxx_g = pxx[idxg]

                        if np.any(~np.isfinite(pxx_g)) or np.any(pxx_g <= 0):
                            item_log["falhas_intervalo_psd"] += 1
                            continue

                        if freqs_ref is None:
                            freqs_ref = freqs_g
                            psds.append(pxx_g)
                        else:
                            if len(freqs_g) != len(freqs_ref) or not np.allclose(freqs_g, freqs_ref):
                                pxx_g = np.interp(freqs_ref, freqs_g, pxx_g)
                            psds.append(pxx_g)

                        usados += 1

                    item_log["canais"][canal][cond] = {
                        "intervalos_validos": usados,
                        "intervalos_total": 10
                    }

                    if usados < 1:
                        continue

                    psd_med = np.mean(np.stack(psds, axis=0), axis=0)

                    foo = ajustar_fooof_no_psd_medio(freqs_ref, psd_med, FMIN_GLOBAL, FMAX_GLOBAL)
                    if foo is None:
                        item_log["canais"][canal][cond]["fooof_ok"] = False
                        continue

                    item_log["canais"][canal][cond]["fooof_ok"] = bool(foo["ok"])
                    item_log["canais"][canal][cond]["r2"] = float(foo["r2"])
                    item_log["canais"][canal][cond]["err"] = float(foo["err"])

                    resultados_bandas = {}
                    for banda_nome, (bmin, bmax) in BANDAS.items():
                        met = metricas_por_banda(
                            foo["freqs"], foo["orig_micro"], foo["ap_micro"], foo["periodic_micro"],
                            bmin, bmax
                        )
                        resultados_bandas[banda_nome] = met

                        plot_zoom_banda_com_stats(
                            resultado_fooof=foo,
                            banda_nome=banda_nome,
                            canal=canal,
                            cond=cond,
                            paciente_id=paciente_id,
                            pasta_saida=PASTA_SAIDA,                
                            fmin_b=bmin,
                            fmax_b=bmax,
                            n_validos=usados,
                            n_total=10
                        )

                    salvar_txt_resumo(
                        pasta_saida=PASTA_SAIDA,                      
                        paciente_id=paciente_id,
                        canal=canal,
                        cond=cond,
                        n_validos=usados,
                        resultados_por_banda=resultados_bandas,
                        fooof_info=foo
                    )

        except Exception as e:
            item_log["processado"] = False
            item_log["erro"] = str(e)
            print(f"❌ Erro em {nome_arquivo}: {e}")

        log_geral["arquivos_processados"].append(item_log)

    caminho_log = os.path.join(PASTA_SAIDA, f"ANALISE_LOG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")  # <- ALTERADO
    salvar_json(caminho_log, log_geral)

    print("\n" + "=" * 80)
    print("✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"📁 Saída: {PASTA_SAIDA}")                                    # <- ALTERADO
    print(f"🧾 Log:  {caminho_log}")
    print("=" * 80)
