const API_BASE = window.UNILEX_API_BASE || "";
const $ = id => document.getElementById(id);
const searchInput = $("search-input"), searchBtn = $("search-btn");
const searchStatus = $("search-status"), resultsEl = $("results");
const lectureInput = $("lecture-input"), extractBtn = $("extract-btn"), extractResultsEl = $("extract-results");
let searchController, extractController, searchVersion = 0, extractVersion = 0;
function getSearchMode() { return document.querySelector('input[name="mode"]:checked').value; }
function escapeHtml(value) { const el = document.createElement("div"); el.textContent = value == null ? "" : String(value); return el.innerHTML; }
function renderEntry(entry) {
  const related = [...new Set(Array.isArray(entry.related_terms) ? entry.related_terms : [])]
    .map(t => `<button type="button" class="related-chip" data-term="${escapeHtml(t).replaceAll('"', '&quot;')}">${escapeHtml(t)}</button>`).join("");
  const source = String(entry.source || "");
  let sourceHtml = escapeHtml(source);
  try { const url = new URL(source); if (["https:", "http:"].includes(url.protocol)) sourceHtml = `<a href="${escapeHtml(url.href).replaceAll('"', '&quot;')}" target="_blank" rel="noopener noreferrer">View source</a>`; } catch (_) {}
  return `<article class="entry-card"><h3>${escapeHtml(entry.term)}</h3>
    <div class="entry-meta"><span class="badge">${escapeHtml(entry.category)}</span><span class="badge">${escapeHtml(entry.difficulty)}</span></div>
    <div class="entry-section"><div class="label">Student-friendly explanation</div><div>${escapeHtml(entry.student_friendly_explanation)}</div></div>
    <div class="entry-section"><div class="label">Formal definition</div><div>${escapeHtml(entry.formal_definition)}</div></div>
    ${entry.example ? `<div class="entry-section"><div class="label">Example</div><div>${escapeHtml(entry.example)}</div></div>` : ""}
    ${related ? `<div class="entry-section"><div class="label">Related terms</div><div class="related-terms">${related}</div></div>` : ""}
    ${source ? `<div class="entry-section source"><div class="label">Source</div>${sourceHtml}</div>` : ""}</article>`;
}
function setSearchStatus(message, error = false) { searchStatus.textContent = message; searchStatus.classList.toggle("error", error); }
async function request(path, options, controller) {
  const timer = setTimeout(() => controller.abort("timeout"), 90000);
  try {
    const response = await fetch(`${API_BASE}${path}`, {...options, signal: controller.signal});
    if (!response.ok) {
      if (response.status === 503) throw new Error("Smart search is temporarily unavailable. Please try Keyword search.");
      if (response.status === 422) throw new Error("Please check the length and format of your input.");
      throw new Error("UniLex could not complete the request. Please try again.");
    }
    return await response.json();
  } finally { clearTimeout(timer); }
}
function failureMessage(error, controller) {
  if (controller.signal.reason === "timeout") return "The request took too long. Please try again, or use Keyword search.";
  if (error instanceof TypeError || error instanceof SyntaxError) return "Could not connect to UniLex. Check your connection and try again.";
  return error.message || "Something went wrong. Please try again.";
}
async function runSearch() {
  const version = ++searchVersion;
  searchController?.abort();
  const controller = searchController = new AbortController();
  const query = searchInput.value.trim();
  resultsEl.innerHTML = ""; searchBtn.disabled = false; resultsEl.setAttribute("aria-busy", "false");
  if (!query) { setSearchStatus("Enter a term or a question to search."); searchInput.focus(); return; }
  if (query.length > 1000) { setSearchStatus("Please use at most 1,000 characters.", true); return; }
  const mode = getSearchMode();
  const endpoint = mode === "semantic" ? "/search/semantic" : "/search";
  setSearchStatus(mode === "semantic" ? "Searching by meaning… This can take a moment." : "Searching…");
  searchBtn.disabled = true; resultsEl.setAttribute("aria-busy", "true");
  try {
    const data = await request(`${endpoint}?q=${encodeURIComponent(query)}&top_k=5`, {}, controller);
    if (version !== searchVersion) return;
    if (!Array.isArray(data.results)) throw new Error("UniLex returned an unexpected response. Please try again.");
    setSearchStatus(`${data.results.length} result(s) — ${mode === "semantic" ? "smart" : "keyword"} search`);
    resultsEl.innerHTML = data.results.length ? data.results.map(renderEntry).join("") : '<p class="status">No matching terms found. Try a CS or AI term, or rephrase your question.</p>';
  } catch (error) {
    if (version === searchVersion) setSearchStatus(failureMessage(error, controller), true);
  } finally {
    if (version === searchVersion) { searchBtn.disabled = false; resultsEl.setAttribute("aria-busy", "false"); }
  }
}
async function runExtract() {
  const version = ++extractVersion;
  extractController?.abort();
  const controller = extractController = new AbortController();
  const text = lectureInput.value.trim();
  extractBtn.disabled = false; extractResultsEl.setAttribute("aria-busy", "false");
  if (!text) { extractResultsEl.innerHTML = '<p class="status">Paste some lecture text first.</p>'; lectureInput.focus(); return; }
  if (text.length > 20000) { extractResultsEl.innerHTML = '<p class="status error">Please use at most 20,000 characters.</p>'; return; }
  extractResultsEl.innerHTML = '<p class="status">Finding terms…</p>';
  extractBtn.disabled = true; extractResultsEl.setAttribute("aria-busy", "true");
  try {
    const data = await request('/extract', {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text})}, controller);
    if (version !== extractVersion) return;
    if (!Array.isArray(data.terms_found)) throw new Error("UniLex returned an unexpected response. Please try again.");
    extractResultsEl.innerHTML = data.terms_found.length ? `<p class="status">${data.terms_found.length} term(s) found</p>` + data.terms_found.map(renderEntry).join("") : '<p class="status">No known UniLex terms found in that text.</p>';
  } catch (error) {
    if (version === extractVersion) extractResultsEl.innerHTML = `<p class="status error">${escapeHtml(failureMessage(error, controller))}</p>`;
  } finally {
    if (version === extractVersion) { extractBtn.disabled = false; extractResultsEl.setAttribute("aria-busy", "false"); }
  }
}
searchBtn.addEventListener("click", runSearch);
searchInput.addEventListener("keydown", event => { if (event.key === "Enter") runSearch(); });
extractBtn.addEventListener("click", runExtract);
document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener("change", () => { if (searchInput.value.trim()) runSearch(); }));
document.addEventListener("click", event => {
  const chip = event.target.closest("button[data-term]");
  if (!chip) return;
  searchInput.value = chip.dataset.term;
  document.querySelector('input[name="mode"][value="tfidf"]').checked = true;
  runSearch(); searchInput.scrollIntoView({behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: "center"});
});
