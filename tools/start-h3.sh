#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Развёртывание MiniMax H3 на чистом поде RunPod с нуля.
# Рассчитан на схему «ничего не храним между сессиями»: ни тома, ни пода.
#
#   bash старт-h3.sh
#
# Кастом-ноды НЕ нужны: все ноды нашего воркфлоу встроены в ComfyUI 0.30+.
# Модели качаем на диск контейнера — Xet ломается при записи на сетевой том
# (проверено 11.08.2026: Internal Writer Error на файлах от 21 ГБ).
#
# Агентство ПИТАНИЛ, 11.08.2026
# ---------------------------------------------------------------------------
set -uo pipefail
START=$SECONDS

# --- ComfyUI ---------------------------------------------------------------
COMFY=""
for c in /workspace/runpod-slim/ComfyUI /workspace/ComfyUI /opt/ComfyUI /ComfyUI; do
    [ -f "$c/main.py" ] && COMFY="$c" && break
done
[ -z "$COMFY" ] && COMFY="$(find / -maxdepth 5 -name main.py -path '*ComfyUI*' 2>/dev/null | head -1 | xargs -r dirname)"
if [ -z "$COMFY" ]; then
    echo "ОШИБКА: ComfyUI не найден — под поднят не из шаблона ComfyUI." >&2; exit 1
fi
echo "==> ComfyUI: $COMFY"

VER=$(grep -o '"[0-9.]*"' "$COMFY/comfyui_version.py" 2>/dev/null | tr -d '"' | head -1)
echo "==> Версия: ${VER:-неизвестна}"
case "${VER%%.*}.$(echo "$VER" | cut -d. -f2)" in
    0.3[0-9]|0.[4-9]*|[1-9]*) : ;;
    *) echo "!! Версия старше 0.30 — поддержки MiniMax H3 может не быть."
       echo "!! Обновить:  cd $COMFY && git pull && pip install -r requirements.txt" ;;
esac

MODELS="$COMFY/models"
mkdir -p "$MODELS"/{diffusion_models,text_encoders,vae}

# --- место -----------------------------------------------------------------
FREE=$(df -BG --output=avail "$MODELS" | tail -1 | tr -dc '0-9')
echo "==> Свободно: ${FREE} ГБ (нужно ~60)"
[ "${FREE:-0}" -lt 65 ] && echo "!! МАЛО МЕСТА. Поду нужен Container Disk от 120 ГБ."

# --- comfy_kitchen: без него падает CLIPLoader ------------------------------
# 22.08.2026: все 63 задания упали с "'NoneType' object has no attribute 'Params'".
# Причина — comfy_kitchen 0.2.10 в образе не умеет fp8/fp4-веса. Лечится обновлением.
KVER=$(python3 -c "import comfy_kitchen,sys;print(getattr(comfy_kitchen,'__version__','0'))" 2>/dev/null || echo 0)
echo "==> comfy_kitchen: ${KVER}"
if [ "$(printf '%s\n0.2.31' "$KVER" | sort -V | head -1)" != "0.2.31" ]; then
    echo "  ! старая версия — обновляю (иначе CLIPLoader упадёт на fp8/fp4)"
    pip install -q -U comfy_kitchen 2>&1 | tail -1
    python3 -c "import comfy_kitchen;print('  теперь:', comfy_kitchen.__version__)" 2>/dev/null
else
    echo "  ок, fp8/fp4 поддерживаются"
fi

# --- инструмент скачивания -------------------------------------------------
command -v hf >/dev/null || pip install -q -U "huggingface_hub[hf_transfer]" 2>&1 | tail -1
export HF_XET_HIGH_PERFORMANCE=1

# Качаем во временную папку на локальном диске, затем переносим.
# Если models/ уже на локальном диске — перенос мгновенный.
TMP=/root/_h3dl
mkdir -p "$TMP"

get () {  # get <путь-в-репо> <целевая-папка>
    local path="$1" dir="$2" name="${1##*/}"
    if [ -s "$dir/$name" ]; then echo "  = уже есть: $name"; return 0; fi
    echo "  ↓ $name"
    if hf download Comfy-Org/MiniMax-H3 "$path" --local-dir "$TMP" >/dev/null 2>&1 && [ -s "$TMP/$path" ]; then
        mv "$TMP/$path" "$dir/$name"
    else
        echo "  ! не скачался: $name"
    fi
}

echo ""
echo "==> Модели MiniMax H3 (~59 ГБ)"
get diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors  "$MODELS/diffusion_models"
get text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors       "$MODELS/text_encoders"
get vae/minimax_h3_video_vae_fp16.safetensors                        "$MODELS/vae"
get vae/minimax_h3_audio_vae_fp32.safetensors                        "$MODELS/vae"
# ref2va нужен только для сюжетов «по референсу»; для рентген-фонов не требуется.
if [ "${WITH_REF2VA:-0}" = "1" ]; then
    get diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors "$MODELS/diffusion_models"
fi
rm -rf "$TMP"

# --- NVFP4 требует Blackwell ------------------------------------------------
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo ""
echo "==> GPU: $GPU"
case "$GPU" in
    *PRO*6000*|*5090*|*B200*|*B300*|*Blackwell*) echo "  Blackwell — NVFP4-энкодер считается нативно, всё верно." ;;
    *) echo "  !! НЕ Blackwell. NVFP4-энкодер работать не будет."
       echo "  !! Скачай int8-вариант вместо него:"
       echo "     hf download Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors --local-dir $MODELS" ;;
esac

# --- итог ------------------------------------------------------------------
echo ""
# --- ComfyUI жив? -----------------------------------------------------------
for i in $(seq 1 30); do
    curl -s -m 2 localhost:8188/system_stats >/dev/null 2>&1 && break
    [ "$i" = 30 ] && echo "!! ComfyUI не отвечает на 8188 — проверь supervisorctl status"
    sleep 2
done
curl -s -m 3 localhost:8188/system_stats >/dev/null 2>&1 && echo "==> ComfyUI отвечает на 8188"

echo "=========================================================="
find "$MODELS" -name "*.safetensors" -exec ls -lL {} \; | awk '{printf "%7.1f ГБ  %s\n", $5/1073741824, $9}'
echo ""
echo "Заняло: $((SECONDS-START)) сек"
echo ""
echo "Перезапустить ComfyUI — ТОЛЬКО по PID:"
echo "    kill \$(pgrep -f 'ComfyUI/main.py' | head -1)      # супервизор поднимет сам"
echo "  НЕ делать pkill -f 'python.*main.py': строка совпадает с текстом собственной"
echo "  команды в SSH-сессии, и ты убиваешь сам себя (проверено дважды, 11 и 22.08)."
echo ""
echo "UI: порт 8188. Задания слать POST на /prompt — см. очередь-угадай-нутриент.py"
echo "=========================================================="
