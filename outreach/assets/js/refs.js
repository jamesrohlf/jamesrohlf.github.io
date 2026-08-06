/* Reference renderer.
 *
 * Reads assets/data/refs.json and fills any element like:
 *
 *   <div class="ref-list" data-topic="lya-pipeline"></div>
 *   <div class="ref-list" data-group="section" data-filters="true"></div>
 *   <div class="ref-list" data-ids="dey2019-legacy,farr2020-qso-strategies"></div>
 *
 *   data-topic   render every ref carrying this topic, in file order
 *   data-ids     render exactly these ids, in the order given
 *   data-group   group output under <h2> by this field (e.g. "section")
 *   data-filters add topic filter buttons above the list
 *   data-sort    "year-desc" to sort newest first (default: file order)
 *
 * Add a paper to refs.json once; every page that asks for it updates.
 */
(function () {
  "use strict";

  var TOPIC_LABELS = {
    "instrument":   "Instrument",
    "data":         "Data & pipeline",
    "cosmology":    "Cosmology",
    "lya":          "Lyman-α",
    "lya-pipeline": "Lyman-α pipeline"
  };

  // Topics used only to build a specific page's list, never offered as a
  // filter button (they would be meaningless on a general publication list).
  var HIDDEN_TOPICS = ["rohlf", "lya-dr1-bao", "lya-dr1-fs",
                       "lya-dr2-bao", "lya-dr2-fs", "lya-p1d",
                       "gq-bao", "gq-fs", "cpe"];

  // BU group members whose ORCID iD is shown wherever they appear in an
  // author list. Add a surname here and it is marked up everywhere at once.
  var BU_AUTHORS = [
    { surname: "Ahlen", orcid: "0000-0001-6098-7247", name: "Steve Ahlen" },
    { surname: "Lodha", orcid: "0009-0004-2558-5655", name: "Kushal Lodha" },
    { surname: "Rohlf", orcid: "0000-0001-6423-9799", name: "James Rohlf" }
  ];

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Append an ORCID mark after each BU author in an already-escaped string.
  // Matches the surname together with any leading initials ("J. W. Rohlf").
  function markBUAuthors(escaped) {
    BU_AUTHORS.forEach(function (a) {
      var re = new RegExp("((?:[A-Z]\\.\\s*)*" + a.surname + ")(?![a-z])", "g");
      escaped = escaped.replace(re, function (match) {
        return match +
          '<a class="orcid" href="https://orcid.org/' + a.orcid + '"' +
          ' target="_blank" rel="noopener"' +
          ' aria-label="ORCID iD for ' + a.name + ': ' + a.orcid + '"' +
          ' title="ORCID ' + a.orcid + '">' +
          '<svg width="11" height="11"><use href="#orcid-mark"></use></svg></a>';
      });
    });
    return escaped;
  }

  function linkPill(href, label) {
    var external = /^https?:/.test(href);
    return '<a href="' + esc(href) + '"' +
           (external ? ' target="_blank" rel="noopener"' : "") +
           ">" + esc(label) + "</a>";
  }

  function refHTML(ref) {
    var links = [];
    if (ref.pdf)   links.push(linkPill(ref.pdf, "PDF"));
    if (ref.doi)   links.push(linkPill("https://doi.org/" + ref.doi, "DOI"));
    if (ref.arxiv) links.push(linkPill("https://arxiv.org/abs/" + ref.arxiv, "arXiv"));
    (ref.extras || []).forEach(function (x) { links.push(linkPill(x.href, x.label)); });

    var journal = [ref.journal, ref.year].filter(Boolean).join(", ");

    return '<div class="ref" id="ref-' + esc(ref.id) + '">' +
             '<p class="ref__title">' + esc(ref.title) + "</p>" +
             '<p class="ref__authors">' + markBUAuthors(esc(ref.authors)) + "</p>" +
             '<p class="ref__journal">' + esc(journal) + "</p>" +
             '<div class="ref__links">' + links.join("") + "</div>" +
           "</div>";
  }

  function select(all, el) {
    var ids   = el.getAttribute("data-ids");
    var topic = el.getAttribute("data-topic");
    var list;

    if (ids) {
      var order = ids.split(",").map(function (s) { return s.trim(); });
      list = order
        .map(function (id) {
          var hit = all.filter(function (r) { return r.id === id; })[0];
          if (!hit) console.warn("refs.js: no reference with id '" + id + "'");
          return hit;
        })
        .filter(Boolean);
    } else if (topic) {
      list = all.filter(function (r) { return (r.topics || []).indexOf(topic) !== -1; });
    } else {
      list = all.slice();
    }

    var sort = el.getAttribute("data-sort");
    if (sort === "year-desc") {
      list = list.slice().sort(function (a, b) { return (b.year || 0) - (a.year || 0); });
    } else if (sort === "date-desc") {
      // Uses the "date" field (arXiv appearance, YYYY-MM-DD). Entries without
      // one fall back to their year so they still land in the right block.
      var key = function (r) { return r.date || (r.year ? r.year + "-00-00" : ""); };
      list = list.slice().sort(function (a, b) { return key(b).localeCompare(key(a)); });
    }
    return list;
  }

  function renderInto(el, list) {
    var groupBy = el.getAttribute("data-group");
    var body = el.querySelector(".ref-list__body") || el;

    if (!list.length) {
      body.innerHTML = '<p class="muted">No references yet.</p>';
      return;
    }
    if (!groupBy) {
      body.innerHTML = list.map(refHTML).join("");
      return;
    }

    var order = [], groups = {};
    list.forEach(function (r) {
      var key = r[groupBy] || "Other";
      if (!groups[key]) { groups[key] = []; order.push(key); }
      groups[key].push(r);
    });
    body.innerHTML = order.map(function (key) {
      return '<h2 class="section-title">' + esc(key) + "</h2>" +
             groups[key].map(refHTML).join("");
    }).join("");
  }

  function addFilters(el, all, baseList) {
    var topics = [];
    baseList.forEach(function (r) {
      (r.topics || []).forEach(function (t) {
        if (topics.indexOf(t) === -1 && HIDDEN_TOPICS.indexOf(t) === -1) topics.push(t);
      });
    });
    if (topics.length < 2) return;

    var bar = document.createElement("div");
    bar.className = "ref-toolbar";
    bar.innerHTML =
      '<button class="ref-filter" data-filter="all" aria-pressed="true">All</button>' +
      topics.map(function (t) {
        return '<button class="ref-filter" data-filter="' + esc(t) + '" aria-pressed="false">' +
               esc(TOPIC_LABELS[t] || t) + "</button>";
      }).join("") +
      '<span class="ref-count"></span>';

    var body = document.createElement("div");
    body.className = "ref-list__body";
    el.textContent = "";
    el.appendChild(bar);
    el.appendChild(body);

    function apply(filter) {
      var shown = filter === "all"
        ? baseList
        : baseList.filter(function (r) { return (r.topics || []).indexOf(filter) !== -1; });
      renderInto(el, shown);
      bar.querySelector(".ref-count").textContent =
        shown.length + (shown.length === 1 ? " paper" : " papers");
      Array.prototype.forEach.call(bar.querySelectorAll(".ref-filter"), function (b) {
        b.setAttribute("aria-pressed", String(b.getAttribute("data-filter") === filter));
      });
    }

    bar.addEventListener("click", function (e) {
      var btn = e.target.closest(".ref-filter");
      if (btn) apply(btn.getAttribute("data-filter"));
    });

    apply("all");
  }

  function boot(data) {
    var all = data.refs || [];
    Array.prototype.forEach.call(document.querySelectorAll(".ref-list"), function (el) {
      var list = select(all, el);
      if (el.getAttribute("data-filters") === "true") {
        addFilters(el, all, list);
      } else {
        renderInto(el, list);
      }
    });
  }

  function fail(err) {
    Array.prototype.forEach.call(document.querySelectorAll(".ref-list"), function (el) {
      el.innerHTML = '<p class="muted">Reference list could not be loaded. ' +
        "(This page must be served over http:// — opening the file directly blocks the data fetch.)</p>";
    });
    console.error("refs.js:", err);
  }

  function start() {
    fetch("assets/data/refs.json", { cache: "no-cache" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(boot)
      .catch(fail);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
