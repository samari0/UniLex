"""Chromium frontend tests using controlled API fixtures, not production accuracy evidence."""
import asyncio,json,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from urllib.parse import urlparse,parse_qs
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
ENTRY=dict(term='Overfitting',category='Machine Learning',difficulty='beginner',formal_definition='A model fits training-specific noise and performs poorly on new data.',student_friendly_explanation='The model remembers its practice examples too closely and struggles with new examples.',example='It scores well on training images but poorly on new photos.',related_terms=['Regularization'],source='https://scikit-learn.org/stable/common_pitfalls.html')
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox']);page=await browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  async def route_api(route):
   req=route.request;q=parse_qs(urlparse(req.url).query).get('q',[''])[0]
   if q=='slow':await asyncio.sleep(.5)
   if q=='unavailable':return await route.fulfill(status=503,json={'detail':'internal'})
   if q=='network':return await route.abort()
   if q=='broken':return await route.fulfill(status=200,body='not-json')
   if '/extract' in req.url:return await route.fulfill(json={'terms_found':[] if json.loads(req.post_data)['text']=='nothing' else [ENTRY]})
   if q=='xyzabc123':return await route.fulfill(json={'results':[]})
   entry={**ENTRY,'term':q if q in ['slow','second','<img src=x onerror=alert(1)>'] else 'Overfitting'}
   return await route.fulfill(json={'results':[entry]})
  for pattern in ['**/search?*','**/search/semantic?*','**/extract']:await page.route(pattern,route_api)
  await page.goto('http://127.0.0.1:8765')
  await page.click('#search-btn');assert 'Enter a term' in await page.inner_text('#search-status')
  await page.click('#extract-btn');assert 'Paste some lecture text' in await page.inner_text('#extract-results')
  for q,message in [('xyzabc123','No matching'),('unavailable','temporarily unavailable'),('network','Could not connect'),('broken','Could not connect')]:
   await page.fill('#search-input',q);await page.click('#search-btn');await page.wait_for_function('!document.querySelector("#search-btn").disabled');assert message in await page.inner_text('main')
  await page.fill('#search-input','slow');await page.click('#search-btn');assert await page.is_disabled('#search-btn')
  await page.fill('#search-input','second');await page.press('#search-input','Enter');await page.wait_for_function('document.querySelector("#results h3")?.textContent === "second"');await page.wait_for_timeout(600);assert await page.inner_text('#results h3')=='second'
  await page.fill('#search-input','<img src=x onerror=alert(1)>');await page.press('#search-input','Enter');await page.wait_for_function('!document.querySelector("#search-btn").disabled');assert await page.locator('#results img').count()==0
  await page.fill('#search-input','Overfitting');await page.press('#search-input','Enter');await page.wait_for_function('!document.querySelector("#search-btn").disabled')
  await page.check('input[value="semantic"]');await page.wait_for_function('document.querySelector("#search-status").textContent.includes("smart")')
  await page.click('button[data-term="Regularization"]');await page.wait_for_function('!document.querySelector("#search-btn").disabled');assert await page.is_checked('input[value="tfidf"]')
  await page.fill('#lecture-input','nothing');await page.click('#extract-btn');await page.wait_for_function('!document.querySelector("#extract-btn").disabled');assert 'No known' in await page.inner_text('#extract-results')
  await page.fill('#lecture-input','Overfitting occurs during machine learning.');await page.click('#extract-btn');await page.wait_for_function('!document.querySelector("#extract-btn").disabled');assert await page.locator('#extract-results h3').count()==1
  await page.fill('#search-input','Overfitting');await page.press('#search-input','Enter');await page.wait_for_function('!document.querySelector("#search-btn").disabled')
  out=ROOT/'docs/screenshots';out.mkdir(parents=True,exist_ok=True)
  for name,w,h in [('phone',390,844),('small-phone',320,740),('tablet',768,1024),('desktop',1440,1000)]:
   await page.set_viewport_size({'width':w,'height':h});await page.evaluate('window.scrollTo(0,0)');assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),name
   await page.screenshot(path=str(out/f'{name}.png'),full_page=True)
  assert not errors,errors
  await browser.close();print('UI_ACCEPTANCE_PASSED: empty, no result, errors, loading, stale responses, escaping, mode, related terms, extraction, four viewport sizes')
server=ThreadingHTTPServer(('127.0.0.1',8765),partial(SimpleHTTPRequestHandler,directory=str(ROOT/'frontend')))
threading.Thread(target=server.serve_forever,daemon=True).start()
try:asyncio.run(main())
finally:server.shutdown()
