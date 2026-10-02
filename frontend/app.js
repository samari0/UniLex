// Point this at your deployed backend URL once you deploy to Render.
// Left as relative paths here so it also works if the frontend is served
// from the same host as the API (see DEPLOYMENT.md).
const API_BASE = window.UNILEX_API_BASE || "";

const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const searchStatus = document.getElementById("search-status");
const resultsEl = document.getElementById("results");

const lectureInput = document.getElementById("lecture-input");
const extractBtn = document.getElementById("extract-btn");
const extractResultsEl = document.getElementById("extract-results");

function getSearchMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderEntry(entry) {
  const related = (entry.related_terms || [])
    .map((t) => `<span class="related-chip">${escapeHtml(t)}</span>`)
    .join("");

  return `
    <article class="entry-card">
      <h3>${escapeHtml(entry.term)}</h3>
      <div class="entry-meta">
        <span class="badge">${escapeHtml(entry.category)}</span>
        <span class="badge">${escapeHtml(entry.difficulty)}</span>
      </div>

      <div class="entry-section">
        <div class="label">Student-friendly explanation</div>
        <div>${escapeHtml(entry.student_friendly_explanation)}</div>
      </div>

      <div class="entry-section">
        <div class="label">Formal definition</div>
        <div>${escapeHtml(entry.formal_definition)}</div>
      </div>

      ${entry.example ? `
      <div class="entry-section">
        <div class="label">Example</div>
        <div>${escapeHtml(entry.example)}</div>
      </div>` : ""}

      ${related ? `
      <div class="entry-section">
        <div class="label">Related terms</div>
        <div class="related-terms">${related}</div>
      </div>` : ""}
    </article>
  `;
}

function renderEmpty(message) {
  resultsEl.innerHTML = `<p class="status">${escapeHtml(message)}</p>`;
}

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) return;

  const mode = getSearchMode();
  const endpoint = mode === "semantic" ? "/search/semantic" : "/search";

  searchStatus.textContent = "Searching...";
  searchStatus.classList.remove("error");
  resultsEl.innerHTML = "";

  try {
    const res = await fetch(`${API_BASE}${endpoint}?q=${encodeURIComponent(query)}&top_k=5`);
    const data = await res.json();

    if (!res.ok) {
      searchStatus.textContent = data.detail || "Something went wrong.";
      searchStatus.classList.add("error");
      return;
    }

    searchStatus.textContent = `${data.results.length} result(s) — ${mode === "semantic" ? "smart" : "keyword"} search`;

    if (data.results.length === 0) {
      renderEmpty("No matching terms found. Try different words.");
      return;
    }

    resultsEl.innerHTML = data.results.map(renderEntry).join("");
  } catch (err) {
    searchStatus.textContent = "Could not reach the UniLex API. Is the backend running?";
    searchStatus.classList.add("error");
  }
}

async function runExtract() {
  const text = lectureInput.value.trim();
  if (!text) return;

  extractResultsEl.innerHTML = `<p class="status">Scanning text...</p>`;

  try {
    const res = await fetch(`${API_BASE}/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      extractResultsEl.innerHTML = `<p class="status error">${escapeHtml(data.detail || "Something went wrong.")}</p>`;
      return;
    }

    if (data.terms_found.length === 0) {
      extractResultsEl.innerHTML = `<p class="status">No known UniLex terms found in that text.</p>`;
      return;
    }

    extractResultsEl.innerHTML = data.terms_found.map(renderEntry).join("");
  } catch (err) {
    extractResultsEl.innerHTML = `<p class="status error">Could not reach the UniLex API.</p>`;
  }
}

searchBtn.addEventListener("click", runSearch);
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});
extractBtn.addEventListener("click", runExtract);
