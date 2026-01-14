from __future__ import annotations

from pathlib import Path
import sqlite3
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
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # 単語スコア（mode=1..5）
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS word_scores (
            level TEXT NOT NULL,
            headword TEXT NOT NULL,
            mode INTEGER NOT NULL,
            score INTEGER NOT NULL,      -- 0/5/10 or 初期1
            attempts INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (level, headword, mode)
        )
        """
    )

    # 文法「読んだ！」管理
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS grammar_reads (
            level TEXT NOT NULL,
            name TEXT NOT NULL,
            read_count INTEGER NOT NULL,
            last_read_at TEXT NOT NULL,
            PRIMARY KEY (level, name)
        )
        """
    )

    conn.commit()
    conn.close()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_word_mode_score(level: str, headword: str, mode: int) -> tuple[int, int]:
    """(score, attempts) / 無ければ初期(1,0)"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT score, attempts FROM word_scores WHERE level=? AND headword=? AND mode=?",
        (level, headword, mode),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return (1, 0)
    return (int(row[0]), int(row[1]))


def set_word_mode_score(level: str, headword: str, mode: int, score: int) -> None:
    prev_score, prev_attempts = get_word_mode_score(level, headword, mode)
    attempts = prev_attempts + 1

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO word_scores(level, headword, mode, score, attempts, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(level, headword, mode)
        DO UPDATE SET score=excluded.score, attempts=excluded.attempts, updated_at=excluded.updated_at
        """,
        (level, headword, mode, int(score), int(attempts), now_iso()),
    )
    conn.commit()
    conn.close()


def get_word_total_score(level: str, headword: str) -> int:
    # mode 1..5（無いものは1）
    total = 0
    for m in range(1, 6):
        s, _ = get_word_mode_score(level, headword, m)
        total += s
    return total


def get_all_word_totals(level: str, headwords: list[str]) -> dict[str, int]:
    """
    DBからまとめて取得（存在しないmodeは1扱い）
    """
    # 初期値は5（1×5）
    totals = {hw: 5 for hw in headwords}

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT headword, mode, score
        FROM word_scores
        WHERE level=? 
        """,
        (level,),
    )
    rows = cur.fetchall()
    conn.close()

    # まず全部 1×5 としておいて、存在する mode を差し替え
    # ただし、既に初期1が含まれているので「(score - 1)」分だけ加算する
    for hw, mode, score in rows:
        if hw in totals and 1 <= int(mode) <= 5:
            totals[hw] += int(score) - 1

    return totals


def mark_grammar_read(level: str, name: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT read_count FROM grammar_reads WHERE level=? AND name=?",
        (level, name),
    )
    row = cur.fetchone()
    if row is None:
        read_count = 1
        cur.execute(
            "INSERT INTO grammar_reads(level, name, read_count, last_read_at) VALUES (?, ?, ?, ?)",
            (level, name, read_count, now_iso()),
        )
    else:
        read_count = int(row[0]) + 1
        cur.execute(
            "UPDATE grammar_reads SET read_count=?, last_read_at=? WHERE level=? AND name=?",
            (read_count, now_iso(), level, name),
        )
    conn.commit()
    conn.close()


def get_grammar_read_stats(level: str, names: list[str]) -> tuple[int, int]:
    """
    (read_unique_count, total)
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM grammar_reads WHERE level=?", (level,))
    read_names = {r[0] for r in cur.fetchall()}
    conn.close()
    total = len(names)
    read_unique = sum(1 for n in names if n in read_names)
    return read_unique, total


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
st.set_page_config(page_title="CEFR 英語教材", layout="centered")
init_db()

st.title("CEFR レベル別 英語教材（単語テスト + 文法）")

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
                    st.toast("記録しました ✅")
            with c2:
                if st.button("微妙（5）"):
                    set_word_mode_score(level, hw, mode, 5)
                    st.toast("記録しました ✅")
            with c3:
                if st.button("不正解（0）"):
                    set_word_mode_score(level, hw, mode, 0)
                    st.toast("記録しました ✅")
            with c4:
                if st.button("この単語をスキップ（更新なし）"):
                    st.toast("記録しました ✅")

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

        q = st.text_input("検索（タイトルの一部）", "")
        show = dfg
        if q.strip():
            show = dfg[dfg["name"].astype(str).str.contains(q, case=False, na=False)]

        titles = show["name"].astype(str).tolist()
        if not titles:
            st.warning("検索結果がありません。")
            st.stop()

        placeholder = "（項目を選んでください）"
        options = [placeholder] + titles
        selected = st.selectbox("開く項目", options, index=0, key="grammar_index_select")

        if st.button("この項目を開く", type="primary", key="grammar_open_btn"):
            if selected == placeholder:
                st.warning("項目を選んでから「この項目を開く」を押してください。")
            else:
                st.session_state.g_idx = int(dfg.index[dfg["name"].astype(str) == selected][0]) + 1
                st.session_state.g_view = "reader"
                st.rerun()

        st.stop()

    # ----- リーダー -----
    if st.button("⟵ インデックスへ戻る", key="grammar_back_to_index"):
        st.session_state.g_view = "index"
        st.stop()

    n = len(dfg)
    if n == 0:
        st.warning("文法データが空です。CSVを確認してください。")
        st.stop()

    c_prev, _, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("← 戻る", key="grammar_prev"):
            st.session_state.g_idx = max(1, st.session_state.g_idx - 1)
    with c_next:
        if st.button("次へ →", key="grammar_next"):
            st.session_state.g_idx = min(n, st.session_state.g_idx + 1)

    idx1 = st.number_input("項目番号", min_value=1, max_value=n, value=st.session_state.g_idx, step=1, key="grammar_number")
    st.session_state.g_idx = int(idx1)

    st.caption(f"{st.session_state.g_idx} / {n}")
    row = dfg.iloc[st.session_state.g_idx - 1]

    title = str(row.get("name", ""))
    summary = str(row.get("summary", ""))
    explanation = str(row.get("explanation", ""))
    original = str(row.get("original", ""))
    translation = str(row.get("translation", ""))

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

    if st.button("読んだ！", key="grammar_read", type="primary"):
        mark_grammar_read(level_g, title)
        st.toast("記録しました ✅")
