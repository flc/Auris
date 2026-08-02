// ── Pronunciation dictionary editor ───────────────────────────────────────────
//
// A two-column table over the same "written = spoken" line format the engine
// parses, so a dictionary that grows to hundreds of names stays readable:
// filter, add, remove, edit in place. Comment lines are kept with the rule
// that follows them and written back untouched.
//
// Usage:
//   const table = createLexiconTable(container, { value, onChange });
//   table.value();          // serialize back to the stored line format
//   table.setValue(text);   // replace all rules

function parseLexiconText(raw) {
  const rows = [];
  let pendingNotes = [];
  let tail = [];

  for (const line of String(raw || '').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith('#')) {
      pendingNotes.push(trimmed);
      continue;
    }
    const parts = trimmed.split(/\s*(?:=>|->|=|→)\s*/);
    if (parts.length < 2) continue;
    rows.push({
      source: parts[0].trim(),
      spoken: parts.slice(1).join('=').trim(),
      notes: pendingNotes,
    });
    pendingNotes = [];
  }
  // Comments after the last rule have nothing to attach to.
  tail = pendingNotes;
  return { rows, tail };
}

function serializeLexicon(rows, tail) {
  const lines = [];
  for (const row of rows) {
    const source = (row.source || '').trim();
    const spoken = (row.spoken || '').trim();
    if (!source && !spoken) continue;
    lines.push(...(row.notes || []));
    lines.push(`${source} = ${spoken}`);
  }
  lines.push(...(tail || []));
  return lines.join('\n');
}

function createLexiconTable(container, { value = '', onChange = null } = {}) {
  const parsed = parseLexiconText(value);
  let rows = parsed.rows;
  let tail = parsed.tail;
  let filter = '';

  container.classList.add('lexicon-editor');
  container.innerHTML = `
    <div class="lexicon-toolbar">
      <label class="lexicon-filter">
        <span class="lexicon-filter-icon" aria-hidden="true">&#9906;</span>
        <input type="search" class="lexicon-filter-input" placeholder="Filter rules"
               aria-label="Filter pronunciation rules" autocomplete="off" spellcheck="false"
               data-transient="view-only">
      </label>
      <span class="lexicon-count" role="status"></span>
      <button type="button" class="btn btn-sm btn-ghost lexicon-sort"
              title="Sort the rules by their written form">Sort A–Z</button>
      <button type="button" class="btn btn-sm btn-ghost lexicon-add">Add rule</button>
    </div>
    <div class="lexicon-table-wrap">
      <table class="lexicon-table">
        <thead>
          <tr>
            <th scope="col">Written</th>
            <th scope="col">Spoken</th>
            <th scope="col"><span class="visually-hidden">Remove</span></th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
      <p class="lexicon-empty"></p>
    </div>`;

  const tbody      = container.querySelector('tbody');
  const emptyEl    = container.querySelector('.lexicon-empty');
  const countEl    = container.querySelector('.lexicon-count');
  const filterEl   = container.querySelector('.lexicon-filter-input');
  const wrapEl     = container.querySelector('.lexicon-table-wrap');

  function notifyChange() {
    if (onChange) onChange(serializeLexicon(rows, tail));
  }

  function matchesFilter(row) {
    if (!filter) return true;
    const needle = filter.toLowerCase();
    return `${row.source} ${row.spoken}`.toLowerCase().includes(needle);
  }

  // The parser keeps the first rule for a word, so a later duplicate never
  // reaches the model. Say so instead of letting it look active.
  function duplicateIndexes() {
    const seen = new Map();
    const dupes = new Set();
    rows.forEach((row, index) => {
      const key = (row.source || '').trim().toLowerCase();
      if (!key) return;
      if (seen.has(key)) dupes.add(index);
      else seen.set(key, index);
    });
    return dupes;
  }

  // Flag duplicates without rebuilding the table: a re-render would drop the
  // focus the user just moved into the next cell.
  function refreshDuplicates() {
    const dupes = duplicateIndexes();
    for (const tr of tbody.children) {
      const index = Number(tr.querySelector('.lexicon-cell')?.dataset.index);
      const duplicate = dupes.has(index);
      tr.classList.toggle('lexicon-duplicate', duplicate);
      tr.title = duplicate
        ? 'A rule for this word already exists above; only the first one is used.'
        : '';
    }
  }

  function render({ focus = null } = {}) {
    const dupes = duplicateIndexes();
    const visible = rows
      .map((row, index) => ({ row, index }))
      .filter(({ row }) => matchesFilter(row));

    tbody.innerHTML = '';
    for (const { row, index } of visible) {
      const tr = document.createElement('tr');
      if (dupes.has(index)) {
        tr.classList.add('lexicon-duplicate');
        tr.title = 'A rule for this word already exists above; only the first one is used.';
      }
      tr.innerHTML = `
        <td><input class="lexicon-cell" data-field="source" data-index="${index}"
                   aria-label="Written form" autocomplete="off" spellcheck="false"></td>
        <td><input class="lexicon-cell" data-field="spoken" data-index="${index}"
                   aria-label="Spoken form" autocomplete="off" spellcheck="false"></td>
        <td class="lexicon-row-actions">
          <button type="button" class="lexicon-remove" data-index="${index}"
                  aria-label="Remove rule">&times;</button>
        </td>`;
      tr.querySelector('[data-field="source"]').value = row.source || '';
      tr.querySelector('[data-field="spoken"]').value = row.spoken || '';
      tbody.appendChild(tr);
    }

    const total = rows.length;
    countEl.textContent = filter
      ? `${visible.length} of ${total} ${total === 1 ? 'rule' : 'rules'}`
      : `${total} ${total === 1 ? 'rule' : 'rules'}`;

    if (!total) {
      emptyEl.textContent = 'No rules yet. Add the first name that gets read wrong.';
    } else if (!visible.length) {
      emptyEl.textContent = `No rule matches “${filter}”.`;
    } else {
      emptyEl.textContent = '';
    }
    emptyEl.classList.toggle('hidden', Boolean(emptyEl.textContent) === false);

    if (focus !== null) {
      const target = tbody.querySelector(`[data-field="source"][data-index="${focus}"]`);
      if (target) {
        target.focus();
        target.scrollIntoView?.({ block: 'nearest' });
      }
    }
  }

  function addRow() {
    // A new rule must stay visible, so clear a filter it would not match.
    if (filter) {
      filter = '';
      filterEl.value = '';
    }
    rows.push({ source: '', spoken: '', notes: [] });
    render({ focus: rows.length - 1 });
    notifyChange();
  }

  container.querySelector('.lexicon-add').addEventListener('click', addRow);

  container.querySelector('.lexicon-sort').addEventListener('click', () => {
    // Accent-aware and stable, so "Álisz" lands with the A's and a duplicate
    // keeps losing to the rule that already won.
    rows.sort((a, b) => (a.source || '').localeCompare(
      b.source || '', undefined, { sensitivity: 'base', numeric: true }));
    render();
    notifyChange();
  });

  filterEl.addEventListener('input', () => {
    filter = filterEl.value.trim();
    render();
  });

  tbody.addEventListener('input', (event) => {
    const cell = event.target.closest('.lexicon-cell');
    if (!cell) return;
    const row = rows[Number(cell.dataset.index)];
    if (!row) return;
    row[cell.dataset.field] = cell.value;
    refreshDuplicates();
    notifyChange();
  });

  tbody.addEventListener('click', (event) => {
    const remove = event.target.closest('.lexicon-remove');
    if (!remove) return;
    rows.splice(Number(remove.dataset.index), 1);
    render();
    notifyChange();
  });

  // A single-line input silently flattens a pasted block of rules into one
  // cell, which turns the rest of them into part of one pronunciation. Expand
  // the paste into rows instead.
  tbody.addEventListener('paste', (event) => {
    const cell = event.target.closest('.lexicon-cell');
    if (!cell) return;
    const text = event.clipboardData?.getData('text/plain') || '';
    if (!/[\r\n]/.test(text)) return;
    const pasted = parseLexiconText(text).rows;
    if (!pasted.length) return;

    event.preventDefault();
    const index = Number(cell.dataset.index);
    const row = rows[index] || {};
    const intoBlankRow = !(row.source || '').trim() && !(row.spoken || '').trim();
    rows.splice(index, intoBlankRow ? 1 : 0, ...pasted);
    if (filter) {
      filter = '';
      filterEl.value = '';
    }
    render({ focus: index + pasted.length - 1 });
    notifyChange();
  });

  tbody.addEventListener('keydown', (event) => {
    const cell = event.target.closest('.lexicon-cell');
    if (!cell || event.key !== 'Enter') return;
    event.preventDefault();
    const index = Number(cell.dataset.index);
    if (cell.dataset.field === 'source') {
      tbody.querySelector(`[data-field="spoken"][data-index="${index}"]`)?.focus();
    } else {
      addRow();
    }
  });

  render();

  return {
    value: () => serializeLexicon(rows, tail),
    setValue: (text) => {
      const next = parseLexiconText(text);
      rows = next.rows;
      tail = next.tail;
      render();
    },
    element: wrapEl,
  };
}

window.createLexiconTable = createLexiconTable;
window.parseLexiconText = parseLexiconText;
window.serializeLexicon = serializeLexicon;
