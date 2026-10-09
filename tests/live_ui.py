"""Real deployed frontend/API smoke test; no mocked routes."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
  page=await browser.new_page();page.set_default_timeout(120000)
  await page.goto('https://unilex-samar.onrender.com/',wait_until='domcontentloaded')
  await page.click('#search-btn');assert 'Enter a term' in await page.inner_text('#search-status')
  await page.click('#extract-btn');assert 'Paste some lecture text' in await page.inner_text('#extract-results')
  await page.fill('#search-input','Overfitting');await page.click('#search-btn');await page.wait_for_function('!document.querySelector("#search-btn").disabled')
  assert await page.locator('#results h3').first.inner_text()=='Overfitting'
  await page.check('input[value="semantic"]');await page.wait_for_function('!document.querySelector("#search-btn").disabled')
  assert 'smart' in await page.inner_text('#search-status')
  await page.fill('#lecture-input','Machine learning uses neural networks. Overfitting can be reduced with regularization.')
  await page.click('#extract-btn');await page.wait_for_function('!document.querySelector("#extract-btn").disabled')
  assert 'Overfitting' in await page.inner_text('#extract-results')
  out=ROOT/'docs/screenshots';out.mkdir(exist_ok=True,parents=True)
  for name,w,h in [('live-desktop',1440,1000),('live-phone',390,844)]:
   await page.set_viewport_size({'width':w,'height':h});await page.evaluate('window.scrollTo(0,0)')
   assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
   await page.screenshot(path=str(out/f'{name}.png'),full_page=True)
  await page.fill('#search-input','xyzabc123');await page.click('#search-btn');await page.wait_for_function('!document.querySelector("#search-btn").disabled');assert 'No matching' in await page.inner_text('#results')
  await browser.close();print('LIVE_UI_PASSED: blank, both modes, extraction, no result, desktop and phone')
asyncio.run(main())
