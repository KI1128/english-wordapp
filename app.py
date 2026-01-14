from __future__ import annotations

import json
from pathlib import Path
import random
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =============================
# パス設定（app.pyと同じフォルダにCSVを置く想定）
# =============================
APP_DIR = Path(__file__).resolve().parent

WORD_FILES = {
    "A1": APP_DIR / "CEFR-J A1（入門）.csv",
    "A2": APP_DIR / "CEFR-J A2（基礎）.csv",
    "B1": APP_DIR / "CEFR-J B1（中級）.csv",
    "B2": APP_DIR / "CEFR-J B2（準上級）.csv",
}

GRAMMAR_FILES = {
    "A1": APP_DIR / "英和辞書_grammars_a1.csv",
    "A2": APP_DIR / "英和辞書_grammars_a2.csv",
    "B1": APP_DIR / "英和辞書_grammars_b1.csv",
    "B2": APP_DIR / "英和辞書_grammars_b2.csv",
}

DB_PATH = APP_DIR / "progress.db"


# =============================
# DB
# =============================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def get_word_mode_score(level: str, headword: str, mode: int) -> tuple[int, int]:
    """
    (score, attempts)
    未挑戦は仕様どおり score=1, attempts=0
    """
    ensure_progress()
    key = f"{level}|{headword}|{mode}"
    rec = st.session_state.progress["word_scores"].get(key)
    if rec is None:
        return (1, 0)
    return (int(rec.get("score", 1)), int(rec.get("attempts", 0)))


def set_word_mode_score(level: str, headword: str, mode: int, score: int) -> None:
    ensure_progress()
    key = f"{level}|{headword}|{mode}"
    prev_score, prev_attempts = get_word_mode_score(level, headword, mode)
    st.session_state.progress["word_scores"][key] = {
        "score": int(score),
        "attempts": int(prev_attempts + 1),
        "updated_at": now_iso(),
    }


def get_all_word_totals(level: str, headwords: list[str]) -> dict[str, int]:
    """
    各単語の「5指標合計」を返す。
    未挑戦は各mode=1なので合計5。
    """
    ensure_progress()
    totals = {hw: 5 for hw in headwords}

    # ある分だけ (score-1) を足す（初期1は既に合計5に含めているため）
    for key, rec in st.session_state.progress["word_scores"].items():
        try:
            lvl, hw, mode_s = key.split("|", 2)
            mode = int(mode_s)
        except Exception:
            continue

        if lvl != level:
            continue
        if hw not in totals:
            continue
        if not (1 <= mode <= 5):
            continue

        score = int(rec.get("score", 1))
        totals[hw] += (score - 1)

    return totals


def mark_grammar_read(level: str, name: str) -> None:
    ensure_progress()
    key = f"{level}|{name}"
    rec = st.session_state.progress["grammar_reads"].get(key)
    if rec is None:
        st.session_state.progress["grammar_reads"][key] = {
            "read_count": 1,
            "last_read_at": now_iso(),
        }
    else:
        st.session_state.progress["grammar_reads"][key] = {
            "read_count": int(rec.get("read_count", 0)) + 1,
            "last_read_at": now_iso(),
        }


def get_grammar_read_stats(level: str, names: list[str]) -> tuple[int, int]:
    ensure_progress()
    read_keys = st.session_state.progress["grammar_reads"].keys()
    read_names = {k.split("|", 1)[1] for k in read_keys if k.startswith(level + "|")}
    total = len(names)
    read_unique = sum(1 for n in names if n in read_names)
    return read_unique, total

def ensure_progress():
    if "progress" not in st.session_state:
        st.session_state.progress = {
            "word_scores": {},     # key: "level|headword|mode" -> {score, attempts, updated_at}
            "grammar_reads": {},   # key: "level|name" -> {read_count, last_read_at}
        }

def export_progress_json() -> str:
    ensure_progress()
    payload = {
        "version": 1,
        "exported_at": now_iso(),
        "word_scores": st.session_state.progress["word_scores"],
        "grammar_reads": st.session_state.progress["grammar_reads"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def import_progress_json(text: str) -> None:
    ensure_progress()
    data = json.loads(text)

    # ゆるめのバリデーション
    if "word_scores" not in data or "grammar_reads" not in data:
        raise ValueError("progress.jsonの形式が違います（word_scores/grammar_readsが見つかりません）")

    st.session_state.progress["word_scores"] = dict(data["word_scores"])
    st.session_state.progress["grammar_reads"] = dict(data["grammar_reads"])



# =============================
# CSVロード
# =============================
def must_exist(path: Path) -> None:
    if not path.exists():
        st.error(f"ファイルが見つかりません: {path}")
        st.stop()


@st.cache_data
def load_words(level: str) -> pd.DataFrame:
    path = WORD_FILES[level]
    must_exist(path)
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")

    # よくある列名に寄せる（存在しない列は空で作る）
    # ここが“保険”。あなたのCSV列が多少違っても動くようにする。
    candidates = {
        "headword": ["headword", "word", "lemma"],
        "pos": ["pos", "part_of_speech"],
        "meaning_ja": ["meaning_ja", "meaning", "ja", "japanese"],
        "ipa": ["ipa", "pronunciation"],
        "example_sentence": ["example_sentence", "example", "sentence_en", "en_sentence"],
        "translated_sentence": ["translated_sentence", "translation", "sentence_ja", "ja_sentence"],
    }

    def pick(colkey: str) -> str:
        for c in candidates[colkey]:
            if c in df.columns:
                return c
        return ""

    # 標準列名へ正規化したDataFrameを返す
    out = pd.DataFrame()
    for key in candidates.keys():
        src = pick(key)
        out[key] = df[src] if src else ""

    # headwordが空の行を除外
    out = out[out["headword"].astype(str).str.strip() != ""].reset_index(drop=True)
    return out


@st.cache_data
def load_grammars(level: str) -> pd.DataFrame:
    path = GRAMMAR_FILES[level]
    must_exist(path)
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")

    candidates = {
        "name": ["name", "title"],
        "summary": ["summary"],
        "explanation": ["explanation", "detail", "description"],
        "original": ["original", "example", "en"],
        "translation": ["translation", "ja"],
    }

    def pick(colkey: str) -> str:
        for c in candidates[colkey]:
            if c in df.columns:
                return c
        return ""

    out = pd.DataFrame()
    for key in candidates.keys():
        src = pick(key)
        out[key] = df[src] if src else ""

    out = out[out["name"].astype(str).str.strip() != ""].reset_index(drop=True)
    return out


# =============================
# 出題ロジック
# =============================
MODE_LABELS = {
    1: "① 英単語 → 日本語（単語/意味）",
    2: "② 日本語（単語/意味） → 英単語",
    3: "③ 英例文 → 日本例文",
    4: "④ 日本例文 → 英例文",
    5: "⑤ リスニング（英）→ 英文＆日本文（※ブラウザ音声）",
}

def choose_word_weighted(level: str, df_words: pd.DataFrame) -> int:
    """
    (51 - 単語の5指標合計) を重みにして単語indexを選ぶ
    """
    headwords = df_words["headword"].astype(str).tolist()
    totals = get_all_word_totals(level, headwords)  # 各単語の5指標合計（初期5）

    weights = []
    for hw in headwords:
        total = totals.get(hw, 5)
        w = 51 - total  # 仕様の式
        if w < 0:
            w = 0
        weights.append(w)

    # 全部0になった場合は均等
    if sum(weights) == 0:
        return random.randrange(len(df_words))

    idx = random.choices(range(len(df_words)), weights=weights, k=1)[0]
    return int(idx)


def pick_mode(enabled_modes: list[int]) -> int:
    return random.choice(enabled_modes)


def speak_button(text: str, button_label: str = "▶ 再生") -> None:
    """
    ブラウザ側の SpeechSynthesis を使って英語TTS（オフラインでも動く環境が多い）
    """
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("\n", " ")
    html = f"""
    <button onclick="
      const u = new SpeechSynthesisUtterance(`{safe}`);
      u.lang = 'en-US';
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    " style="padding:8px 12px; border-radius:8px; border:1px solid #ccc; cursor:pointer;">
      {button_label}
    </button>
    """
    components.html(html, height=60)


# =============================
# UI
# =============================
def flash(msg: str):
    st.session_state["_flash"] = msg

def show_flash():
    msg = st.session_state.pop("_flash", None)
    if msg:
        st.success(msg)  # これなら画面に残る（toastより確実）

st.set_page_config(page_title="CEFR 英語教材", layout="centered")

st.title("CEFR レベル別 英語教材（単語テスト + 文法）")

show_flash()
ensure_progress()

if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0

with st.sidebar:
    st.header("進捗（読み込み/書き出し）")

    up = st.file_uploader(
        "progress.json を選択",
        type=["json"],
        key=f"progress_uploader_{st.session_state.uploader_nonce}"
    )

    if st.button("このファイルを読み込む", type="primary"):
        if up is None:
            st.warning("まず progress.json を選択してください。")
        else:
            try:
                import_progress_json(up.read().decode("utf-8"))
                flash("進捗を読み込みました ✅")
                # uploaderを強制的に空にする（×を押さなくてよくなる）
                st.session_state.uploader_nonce += 1
                st.rerun()
            except Exception as e:
                st.error(f"読み込み失敗: {e}")

    data = export_progress_json()
    st.download_button(
        "進捗を書き出す（progress.json）",
        data=data,
        file_name="progress.json",
        mime="application/json",
    )

tab_words, tab_grammar = st.tabs(["🧠 単語テスト", "📘 文法"])


# -----------------------------
# 単語テスト
# -----------------------------
with tab_words:
    level = st.selectbox("レベル", ["A1", "A2", "B1", "B2"], key="word_level")
    dfw = load_words(level)
    st.caption(f"単語数: {len(dfw)}")

    # --- 出題設定（起動直後は問題を出さない） ---
    st.markdown("### 出題設定")
    include_listening = st.checkbox("⑤ リスニングも含める", value=False)
    enabled_modes = [1, 2, 3, 4] + ([5] if include_listening else [])

    # --- セッション状態 ---
    if "started" not in st.session_state:
        st.session_state.started = False
    if "q_idx" not in st.session_state:
        st.session_state.q_idx = None
    if "q_mode" not in st.session_state:
        st.session_state.q_mode = None
    if "revealed" not in st.session_state:
        st.session_state.revealed = False

    def new_question():
        st.session_state.revealed = False
        st.session_state.q_idx = choose_word_weighted(level, dfw)
        st.session_state.q_mode = pick_mode(enabled_modes)

    # --- まず出題ボタンを押させる ---
    c_start, c_reset = st.columns([1, 1])
    with c_start:
        if st.button("出題", type="primary"):
            st.session_state.started = True
            new_question()
    with c_reset:
        if st.button("出題を終了（リセット）"):
            st.session_state.started = False
            st.session_state.q_idx = None
            st.session_state.q_mode = None
            st.session_state.revealed = False

    # --- 出題開始前はここで止める ---
    if not st.session_state.started or st.session_state.q_idx is None:
        st.info("音声の有無を選んでから「出題」を押してください。")
    else:
        # --- ここから先は「問題表示」 ---
        row = dfw.iloc[int(st.session_state.q_idx)]
        hw = str(row["headword"])
        pos = str(row.get("pos", ""))
        meaning_ja = str(row.get("meaning_ja", ""))
        ipa = str(row.get("ipa", ""))
        ex_en = str(row.get("example_sentence", ""))
        ex_ja = str(row.get("translated_sentence", ""))

        mode = int(st.session_state.q_mode)

        st.divider()
        st.markdown(f"### 問題：{MODE_LABELS[mode]}")

        # 問題提示（答えは出さない）
        if mode == 1:
            st.subheader(hw)
            if ipa:
                st.text(f"IPA: {ipa}")
            if pos:
                st.text(f"品詞: {pos}")

        elif mode == 2:
            st.subheader("（日本語）")
            st.write(meaning_ja)

        elif mode == 3:
            st.subheader("（英例文）")
            st.write(ex_en)

        elif mode == 4:
            st.subheader("（日本例文）")
            st.write(ex_ja)

        elif mode == 5:
            st.subheader("（リスニング）")
            if ex_en.strip():
                speak_button(ex_en, "▶ 英文を再生")
            else:
                st.info("この単語は例文が空なので、リスニング出題が難しいです。別問題に切り替えてください。")

        # 「答えを見る」
        st.markdown("###")
        if st.button("答えを見る", type="primary"):
            st.session_state.revealed = True

        # 全セクション表示（答え）
        if st.session_state.revealed:
            if st.button("次の問題"):
                new_question()
                st.rerun()

            st.success("答え（全セクション）")

            st.markdown("**英単語**")
            st.write(hw)
            if ipa:
                st.markdown("**IPA**")
                st.write(ipa)
            if pos:
                st.markdown("**品詞**")
                st.write(pos)

            st.markdown("**日本語（単語/意味）**")
            st.write(meaning_ja)

            st.markdown("**英例文**")
            st.write(ex_en)

            st.markdown("**日本例文**")
            st.write(ex_ja)

            # 自己評価 → スコア更新
            st.markdown("### 自己評価（このモードのスコアを更新）")
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            with c1:
                if st.button("正解（10）"):
                    set_word_mode_score(level, hw, mode, 10)
                    flash("記録しました ✅")
                    st.rerun()
            with c2:
                if st.button("微妙（5）"):
                    set_word_mode_score(level, hw, mode, 5)
                    flash("記録しました ✅")
                    st.rerun()
            with c3:
                if st.button("不正解（0）"):
                    set_word_mode_score(level, hw, mode, 0)
                    flash("記録しました ✅")
                    st.rerun()
            with c4:
                if st.button("この単語をスキップ（更新なし）"):
                    flash("記録しました ✅")
                    st.rerun()

            # 現在スコアの見える化
            m_scores = []
            for m in range(1, 6):
                s, a = get_word_mode_score(level, hw, m)
                m_scores.append((m, s, a))
            total = sum(s for _, s, _ in m_scores)

            with st.expander("この単語のスコア状況（1〜5）"):
                st.write({f"mode{m}": {"score": s, "attempts": a} for (m, s, a) in m_scores})
                st.write(f"合計（5指標）: {total} / 50（初期は5）")
                st.caption("重み = 51 - 合計（仕様どおり） → 合計が高いほど出にくくなります。")


# -----------------------------
# 文法
# -----------------------------
with tab_grammar:
    level_g = st.selectbox("レベル", ["A1", "A2", "B1", "B2"], key="grammar_level")
    dfg = load_grammars(level_g)
    names = dfg["name"].astype(str).tolist()

    # 状態
    if "g_view" not in st.session_state:
        st.session_state.g_view = "index"  # "index" or "reader"
    if "g_idx" not in st.session_state:
        st.session_state.g_idx = 1

    # 読了統計
    read_unique, total_g = get_grammar_read_stats(level_g, names)
    st.caption(f"文法項目数: {total_g} / 読了（ユニーク）: {read_unique}")

    # ----- インデックス -----
    if st.session_state.g_view == "index":
        st.markdown("### 文法インデックス")

        ensure_progress()
        read_keys = st.session_state.progress["grammar_reads"].keys()
        read_set = {k.split("|", 1)[1] for k in read_keys if k.startswith(level_g + "|")}

        only_unread = st.checkbox("未読のみ表示", value=False)
        q = st.text_input("検索（タイトルの一部）", "")

        show = dfg
        if q.strip():
            show = dfg[dfg["name"].astype(str).str.contains(q, case=False, na=False)]

        titles = show["name"].astype(str).tolist()
        if only_unread:
            titles = [t for t in titles if t not in read_set]

        # 表示用：読了は ✅ を付ける
        label_map = {}
        labeled_titles = []
        for t in titles:
            label = f"✅ {t}" if t in read_set else f"⬜ {t}"
            label_map[label] = t
            labeled_titles.append(label)

        placeholder = "（項目を選んでください）"
        options = [placeholder] + labeled_titles
        selected_label = st.selectbox("開く項目", options, index=0, key="grammar_index_select")

        selected = None if selected_label == placeholder else label_map[selected_label]

        if st.button("この項目を開く", type="primary", key="grammar_open_btn"):
            if selected is None:
                st.warning("項目を選んでから「この項目を開く」を押してください。")
            else:
                new_idx = int(dfg.index[dfg["name"].astype(str) == selected][0]) + 1
                st.session_state.g_idx = new_idx
                st.session_state["grammar_number"] = new_idx
                st.session_state.g_view = "reader"
                st.rerun()

        st.stop()


    # ----- リーダー -----
    if st.button("⟵ インデックスへ戻る", key="grammar_back_to_index"):
        st.session_state.g_view = "index"
        st.rerun()

    n = len(dfg)

    # 初回 or レベル切替時に grammar_number を整える
    if "grammar_number" not in st.session_state:
        st.session_state["grammar_number"] = int(st.session_state.g_idx)

    # 範囲外に落ちないよう保険
    st.session_state["grammar_number"] = max(1, min(n, int(st.session_state["grammar_number"])))

    # ページ送りボタン（grammar_number だけを動かす）
    c_prev, _, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("← 戻る", key="grammar_prev"):
            st.session_state["grammar_number"] = max(1, st.session_state["grammar_number"] - 1)
            st.rerun()
    with c_next:
        if st.button("次へ →", key="grammar_next"):
            st.session_state["grammar_number"] = min(n, st.session_state["grammar_number"] + 1)
            st.rerun()

    # number_input（value=は渡さない / keyで管理）
    st.number_input(
        "項目番号",
        min_value=1, max_value=n,
        step=1,
        key="grammar_number",
    )

    # ★ここで確定：この後は g_idx を読むだけ
    st.session_state.g_idx = int(st.session_state["grammar_number"])

    # この g_idx で row/title を決める（同期後）
    row = dfg.iloc[st.session_state.g_idx - 1]
    title = str(row.get("name", ""))

    ensure_progress()
    gkey = f"{level_g}|{title}"
    rec = st.session_state.progress["grammar_reads"].get(gkey)
    is_read = (rec is not None)

    st.caption(f"{st.session_state.g_idx} / {n}   {'✅読了' if is_read else '⬜未読'}")

    summary = str(row.get("summary", ""))
    explanation = str(row.get("explanation", ""))
    original = str(row.get("original", ""))
    translation = str(row.get("translation", ""))

    if rec:
        st.success(f"✅ 読了（回数: {rec.get('read_count', 1)} / 最終: {rec.get('last_read_at','')}）")
    else:
        st.info("📌 未読")

    st.subheader(title)
    if summary.strip():
        st.info(summary)

    if explanation.strip():
        st.markdown(explanation)
    else:
        st.write("（解説なし）")

    st.markdown("**例文**")
    st.write(original)
    st.write("—")
    st.write(translation)

    # 読んだ！：保存 → その場で rec を引き直して“即反映”
    if st.button("読んだ！", key=f"grammar_read_{level_g}_{st.session_state.g_idx}", type="primary"):
        mark_grammar_read(level_g, title)
        rec = st.session_state.progress["grammar_reads"].get(gkey)
        st.success("記録しました ✅")

