"""
scraper.py

One-time offline script to crawl SHL's product pages and build catalog.json.
Not called at runtime — just run this once to refresh the catalog data.

Usage: python scraper.py

NOTE: SHL's old catalog URL (https://www.shl.com/solutions/products/product-catalog/)
now redirects to a JS-rendered marketing page, not a filterable table.
The actual product pages live under /products/assessments/{category}/{slug}/.
We seed the crawler with known category URLs and follow links from there.

SHL rate-limits bots, so we use Playwright + a realistic user-agent,
and add delays between requests.
"""

import asyncio
import json
import logging
import re

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = "https://www.shl.com"
OUTPUT = "catalog.json"

# these are the category pages we know about — scraped from SHL's nav
CATEGORY_PAGES = [
    "/products/assessments/",
    "/products/assessments/personality-assessment/",
    "/products/assessments/cognitive-assessments/",
    "/products/assessments/behavioral-assessments/",
    "/products/assessments/job-focused-assessments/",
    "/products/assessments/skills-and-simulations/",
    "/products/assessments/skills-and-simulations/call-center-simulations/",
    "/products/assessments/skills-and-simulations/coding-simulations/",
    "/products/assessments/skills-and-simulations/technical-skills/",
    "/products/assessments/skills-and-simulations/language-evaluation/",
    "/products/assessments/assessment-and-development-centers/",
    "/products/360/",
    "/products/video-interviews/",
]

# generic page names we want to skip (hub/category pages, not individual products)
SKIP_NAMES = {
    "shl assessments", "personality assessment", "cognitive assessments",
    "behavioral assessments", "skills and simulations", "job-focused assessments",
    "our products", "products",
}


def infer_test_type(url: str, name: str = "") -> str:
    """Best-effort test type from URL and name. Not perfect but good enough."""
    u = url.lower()
    n = name.lower()
    if "opq" in u or "occupational personality" in n:
        return "Personality & Behavior"
    if "motivation" in u or "motivational" in n:
        return "Personality & Behavior"
    if "verify" in u or "numerical" in n or "verbal" in n or "inductive" in n or "deductive" in n:
        return "Ability & Aptitude"
    if "cognitive" in u:
        return "Ability & Aptitude"
    if "situational" in u or "sjt" in u:
        return "Situational Judgment"
    if "coding" in u:
        return "Technical Skills - Coding"
    if "technical" in u:
        return "Technical Skills"
    if "language" in u or "svar" in u:
        return "Language Skills"
    if "call-center" in u:
        return "Skills & Simulations"
    if "simulation" in u or "business-skills" in u:
        return "Skills & Simulations"
    if "global-skills" in u or "gsa" in u:
        return "Competency & Behavior"
    if "realistic-job" in u:
        return "Job Preview"
    if "universal-competency" in u:
        return "Competency Framework"
    if "job-focused" in u:
        return "Job Focused"
    if "assessment-and-development" in u:
        return "Assessment & Development"
    if "360" in u:
        return "360 Feedback"
    if "video" in u:
        return "Video Interview"
    return "Assessment"


async def get_description(page) -> str:
    """
    Extract page description by skipping cookie/nav/footer elements.
    SHL's cookie consent banner loads early and pollutes paragraph extraction
    if you just grab all <p> tags.
    """
    return await page.evaluate("""
        () => {
            const skippedParents = '#CybotCookiebotDialog, nav, footer, header, [class*="cookie"]';
            const selectors = ['main p', 'article p', 'section p', '.banner p', 'p'];
            for (const sel of selectors) {
                const els = document.querySelectorAll(sel);
                const texts = [];
                for (const el of els) {
                    if (el.closest(skippedParents)) continue;
                    const t = el.innerText.trim();
                    if (t.length > 50 && !t.toLowerCase().includes('cookie') && !t.includes('©')) {
                        texts.push(t);
                        if (texts.length >= 2) break;
                    }
                }
                if (texts.length) return texts.join(' ').slice(0, 500);
            }
            return '';
        }
    """)


async def scrape_page(page, url: str) -> dict | None:
    """Scrape a single product page."""
    try:
        # Changed from networkidle to domcontentloaded to prevent silent timeouts on tracker-heavy pages
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)
    except Exception as e:
        logger.warning("Timeout/error on %s: %s", url, e)
        return None

    title = await page.title()
    h1_el = await page.query_selector("h1")
    h1 = re.sub(r"\s+", " ", await h1_el.inner_text()).strip() if h1_el else ""
    name = h1 or title.split("|")[0].strip()

    # skip hub pages and error pages
    if not name or len(name) < 5:
        return None
    if any(g in name.lower() for g in SKIP_NAMES):
        logger.info("  skip (hub page): %s", name)
        return None
    if "page not found" in name.lower() or "confirm you are human" in name.lower():
        logger.info("  skip (bot block / 404): %s", name)
        return None

    description = await get_description(page)
    body = await page.evaluate("() => document.body.innerText.toLowerCase()")

    # check for duration mentioned on page
    dur_match = re.search(r"(\d+)\s*(minutes?|mins?|hours?|hrs?)", body, re.I)
    duration = dur_match.group(0) if dur_match else ""

    remote = any(kw in body for kw in ["remote testing", "administered remotely", "online proctoring"])
    adaptive = any(kw in body for kw in ["adaptive", "item response theory", "irt"])

    return {
        "name": name,
        "url": url.rstrip("/") + "/",
        "test_type": infer_test_type(url, name),
        "description": description,
        "duration": duration,
        "remote_testing": remote,
        "adaptive_support": adaptive,
    }


async def find_product_links(page, category_url: str) -> list[str]:
    """Visit a category page and collect links to individual product pages."""
    full_url = BASE + category_url
    try:
        await page.goto(full_url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(1.5)
    except Exception:
        return []

    links = await page.evaluate("""
        () => Array.from(document.querySelectorAll('a[href]'))
            .map(a => a.href)
            .filter(h => h.includes('shl.com/products/'))
    """)

    current_depth = len([p for p in category_url.strip("/").split("/") if p])
    results = []
    skip_words = ["rss", "login", "careers", "resources", "privacy", "cookie"]

    for link in links:
        path = link.replace(BASE, "").rstrip("/")
        depth = len([p for p in path.strip("/").split("/") if p])
        if depth > current_depth and not any(s in link.lower() for s in skip_words):
            results.append(link.rstrip("/") + "/")

    return list(dict.fromkeys(results))


async def main():
    logger.info("Starting SHL catalog scrape...")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await ctx.new_page()

        # collect product URLs from all category pages
        all_urls: set[str] = set()
        for cat in CATEGORY_PAGES:
            links = await find_product_links(page, cat)
            for link in links:
                all_urls.add(link)
            await asyncio.sleep(1.0)

        # also add the category pages themselves — some are individual products
        for cat in CATEGORY_PAGES:
            all_urls.add(BASE + cat.rstrip("/") + "/")

        logger.info("Found %d URLs to check", len(all_urls))

        catalog = []
        for i, url in enumerate(sorted(all_urls), 1):
            logger.info("[%d/%d] %s", i, len(all_urls), url)
            item = await scrape_page(page, url)
            if item:
                catalog.append(item)
            await asyncio.sleep(1.2)

        await browser.close()

    # deduplicate
    seen = set()
    deduped = []
    for item in catalog:
        if item["url"] not in seen:
            seen.add(item["url"])
            deduped.append(item)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    logger.info("Done — %d items saved to %s", len(deduped), OUTPUT)
    for item in deduped:
        logger.info("  [%s] %s", item["test_type"], item["name"])


if __name__ == "__main__":
    asyncio.run(main())
