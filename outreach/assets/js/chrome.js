/* Shared page chrome — masthead, navigation, footer.
 *
 * Every page loads this and gets identical chrome. Nothing to copy-paste:
 * edit the NAV array below and all pages update at once.
 * Injected at the start/end of <body>; pages only supply their <main>.
 */
(function () {
  "use strict";

  var NAV = [
    { href: "index.html",          label: "Outreach" },
    { href: "how-desi-works.html", label: "How DESI Works" },
    { href: "https://jamesrohlf.github.io/", label: "Talks & papers" }
  ];

  var SITE_TITLE = "DESI Outreach";
  var TAGLINE    = "Jim Rohlf \u00b7 Boston University";

  function currentPage() {
    var last = location.pathname.split("/").pop();
    return (last === "" ? "index.html" : last).toLowerCase();
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ORCID iD mark, defined once per page. Referenced with <use href="#orcid-mark">
  // by the team list and by refs.js when it marks up BU authors.
  function orcidSymbol() {
    return '' +
      '<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false">' +
        '<symbol id="orcid-mark" viewBox="0 0 256 256">' +
          '<path fill="#A6CE39" d="M256 128c0 70.7-57.3 128-128 128S0 198.7 0 128 57.3 0 128 0s128 57.3 128 128z"/>' +
          '<path fill="#FFF" d="M86.3 186.2H70.9V79.1h15.4v107.1z"/>' +
          '<path fill="#FFF" d="M108.9 79.1h41.6c39.6 0 57 28.3 57 53.6 0 27.5-21.5 53.6-56.8 53.6h-41.8V79.1zm15.4 93.3h24.5c34.9 0 42.9-26.5 42.9-39.7 0-21.5-13.7-39.7-43.7-39.7h-23.7v79.4z"/>' +
          '<path fill="#FFF" d="M88.7 56.8c0 5.5-4.5 10.1-10.1 10.1s-10.1-4.6-10.1-10.1c0-5.6 4.5-10.1 10.1-10.1s10.1 4.5 10.1 10.1z"/>' +
        '</symbol>' +
      '</svg>';
  }

  function masthead() {
    return '' +
      orcidSymbol() +
      '<a class="skip-link" href="#main">Skip to content</a>' +
      '<header class="masthead">' +
        '<div class="masthead__inner">' +
          '<a class="masthead__logo" href="https://www.desi.lbl.gov/">' +
            '<img src="assets/img/desi-logo.png" alt="DESI collaboration">' +
          '</a>' +
          '<p class="masthead__title">' +
            '<a href="index.html">' + esc(SITE_TITLE) + '</a>' +
            '<span class="masthead__tagline">' + esc(TAGLINE) + '</span>' +
          '</p>' +
          '<a class="masthead__bu" href="https://www.bu.edu/physics/">' +
            '<img src="assets/img/bu-physics-logo.png" alt="Boston University Physics">' +
          '</a>' +
        '</div>' +
      '</header>';
  }

  function nav() {
    var here = currentPage();
    var links = NAV.map(function (item) {
      var active = item.href.toLowerCase() === here ? ' aria-current="page"' : "";
      return '<a href="' + item.href + '"' + active + '>' + item.label + "</a>";
    }).join("");
    return '<nav class="sitenav" aria-label="Main"><div class="sitenav__inner">' + links + "</div></nav>";
  }

  function footer() {
    var year = new Date().getFullYear();
    return '' +
      '<footer class="sitefoot">' +
        '<div class="sitefoot__inner">' +
          "<div>" +
            "<p><strong>Jim Rohlf</strong></p>" +
            '<p><a href="https://www.bu.edu/physics/profile/james-rohlf/">BU faculty profile</a></p>' +
            '<p><a href="mailto:rohlf@bu.edu">rohlf@bu.edu</a></p>' +
          "</div>" +
          "<div>" +
            "<p><strong>DESI</strong></p>" +
            '<p><a href="https://www.desi.lbl.gov/">DESI Collaboration</a></p>' +
            '<p><a href="https://data.desi.lbl.gov/doc/">DESI Data Documentation</a></p>' +
          "</div>" +
          "<div>" +
            "<p>&copy; " + year + " James W. Rohlf</p>" +
            '<p class="muted">Dark Energy Spectroscopic Instrument</p>' +
          "</div>" +
        "</div>" +
      "</footer>";
  }

  function render() {
    document.body.insertAdjacentHTML("afterbegin", masthead() + nav());
    document.body.insertAdjacentHTML("beforeend", footer());

    // Reveal anything that only makes sense once scripting is confirmed.
    Array.prototype.forEach.call(document.querySelectorAll(".js-only"), function (el) {
      el.style.display = "";
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
