#!/bin/bash
# Blender port of ue_code/render_single_scene.sh
# scene setup -> construct sequences -> render each sequence (images + vertex data)
set -u

# ---------------------------------------------------------------- common settings
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"   # linux: /opt/blender/blender
ENV_BLEND="${ENV_BLEND:-/path/to/environment.blend}"
SCENE_TYPE="${SCENE_TYPE:-outdoor}"          # outdoor | indoor
FILE_MAP_INDEX="${FILE_MAP_INDEX:-DEBUG}"

CODE_FOLDER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FOLDER="$CODE_FOLDER/logs"
OBJECT_DATA="$CODE_FOLDER/object_data"
BBOX_FOLDER="$OBJECT_DATA/collected_bbox_info"
OBJECT_FILE="$OBJECT_DATA/collected_object.json"
SCENE_SETUP_CONFIG="$CODE_FOLDER/config/template.json"

SCENE_INFO_FOLDER="$LOG_FOLDER/scene_info/$FILE_MAP_INDEX"
SCENE_INFO="$SCENE_INFO_FOLDER/scene_info.json"
STATUS="$SCENE_INFO_FOLDER/status.txt"
SEQUENCE_LOG_FOLDER="$LOG_FOLDER/sequence/$FILE_MAP_INDEX"
PROGRESS_LOG="$LOG_FOLDER/progress_$FILE_MAP_INDEX.txt"
RENDER_LOG="$LOG_FOLDER/render/$FILE_MAP_INDEX.txt"
RENDER_OUTPUT_FOLDER="$CODE_FOLDER/render_output/$FILE_MAP_INDEX"

# ---------------------------------------------------------------- render settings
EXPECT_QUEUE_NUM="${EXPECT_QUEUE_NUM:-1}"
EXPECT_SEQUENCE_NUM="${EXPECT_SEQUENCE_NUM:-3}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
ENGINE="${ENGINE:-CYCLES}"
SAMPLES="${SAMPLES:-64}"
DEVICE="${DEVICE:-GPU}"
MAX_RETRIES="${MAX_RETRIES:-3}"

mkdir -p "$LOG_FOLDER" "$SCENE_INFO_FOLDER" "$SEQUENCE_LOG_FOLDER" "$(dirname "$RENDER_LOG")" "$RENDER_OUTPUT_FOLDER"
touch "$RENDER_LOG"

# ---------------------------------------------------------------- 1. scene setup
echo "$FILE_MAP_INDEX---set up scene----" >> "$PROGRESS_LOG"
if [ ! -f "$STATUS" ] || ! head -n 1 "$STATUS" | grep -q "^SUCCESS"; then
    "$BLENDER" -b "$ENV_BLEND" -P "$CODE_FOLDER/scripts/run_scene_setup.py" -- \
        --config "$SCENE_SETUP_CONFIG" --output "$SCENE_INFO" --status "$STATUS"
fi
if ! head -n 1 "$STATUS" | grep -q "^SUCCESS"; then
    echo "scene setup failed, see $STATUS"; exit 1
fi

# ---------------------------------------------------------------- 2. construct sequences
echo "$FILE_MAP_INDEX---construct scene----" >> "$PROGRESS_LOG"
counter=0
while [ $counter -lt "$EXPECT_QUEUE_NUM" ]; do
    log_file="$SEQUENCE_LOG_FOLDER/QUEUE_$(printf %04d $counter).txt"
    if [ -s "$log_file" ]; then
        echo "queue $counter already constructed"; counter=$((counter + 1)); continue
    fi
    "$BLENDER" -b "$ENV_BLEND" -P "$CODE_FOLDER/scripts/run_construct.py" -- \
        --scene_info "$SCENE_INFO" --object_file "$OBJECT_FILE" --bbox_folder "$BBOX_FOLDER" \
        --anno_base "$RENDER_OUTPUT_FOLDER" --expect_sequence_num "$EXPECT_SEQUENCE_NUM" \
        --group_index "$counter" --sequence_log_folder "$SEQUENCE_LOG_FOLDER" --scene_type "$SCENE_TYPE"
    counter=$((counter + 1))
done

# ---------------------------------------------------------------- 3. render every sequence
echo "$FILE_MAP_INDEX---render----" >> "$PROGRESS_LOG"
for seq_dir in $(find "$RENDER_OUTPUT_FOLDER" -maxdepth 1 -mindepth 1 -type d -name "SEQUENCE_*" | sort); do
    seq_name="$(basename "$seq_dir")"
    if grep -q "^$seq_name$" "$RENDER_LOG"; then
        echo "$seq_name already rendered, skipping"; continue
    fi
    echo "$FILE_MAP_INDEX ---render---- $seq_name" >> "$PROGRESS_LOG"
    retry=0; ok=false
    while [ $retry -lt "$MAX_RETRIES" ] && [ $ok = false ]; do
        "$BLENDER" -b "$ENV_BLEND" -P "$CODE_FOLDER/scripts/run_render.py" -- \
            --sequence_dir "$seq_dir" --width "$WIDTH" --height "$HEIGHT" \
            --engine "$ENGINE" --samples "$SAMPLES" --device "$DEVICE"
        if [ -n "$(ls -A "$seq_dir/images" 2>/dev/null)" ] && [ -f "$seq_dir/faces.bin" ]; then
            ok=true; echo "$seq_name" >> "$RENDER_LOG"; echo "render successful for $seq_name"
        else
            retry=$((retry + 1)); echo "render failed for $seq_name (attempt $retry/$MAX_RETRIES)"; sleep 10
        fi
    done
done
echo "$FILE_MAP_INDEX---done----" >> "$PROGRESS_LOG"
