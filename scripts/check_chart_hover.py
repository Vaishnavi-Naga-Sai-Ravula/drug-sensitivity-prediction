from pathlib import Path
from urllib.parse import urlparse
import argparse
import functools
import http.server
import threading
from playwright.sync_api import sync_playwright
import base64,json,zipfile
from io import BytesIO
import pandas as pd

parser=argparse.ArgumentParser(description='Optional browser regression checks; requires Playwright and Chromium.')
parser.add_argument('--url',default='http://127.0.0.1:8501')
parser.add_argument('--output',type=Path,default=Path('.local_training/hover_browser'))
parser.add_argument('--chromium-path',default='/usr/bin/chromium' if Path('/usr/bin/chromium').exists() else None)
args=parser.parse_args()
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(OUT))
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
REPORT_URL=f'http://127.0.0.1:{server.server_port}/report.html'


def plot_frame(page):
    page.wait_for_function("Array.from(document.querySelectorAll('iframe')).some(frame=>{try{return Boolean(frame.contentWindow.pathdspPlotsReady)}catch(e){return false}})",timeout=120000)
    for frame in page.frames:
        if frame==page.main_frame:continue
        if frame.evaluate('Boolean(window.pathdspPlotsReady)'):
            frame.evaluate('window.pathdspPlotsReady')
            return frame
    raise AssertionError('No interactive chart frame found')


def coordinates(page,frame,name,x,y,subplot='xy'):
    plot=frame.locator('[data-chart="'+name+'"] .plot-area')
    plot.scroll_into_view_if_needed()
    relative=plot.evaluate(r'''(gd,args)=>{const sub=args.subplot.match(/^(x\d*)(y\d*)$/);const ax=gd._fullLayout['xaxis'+sub[1].slice(1)],ay=gd._fullLayout['yaxis'+sub[2].slice(1)];return {x:ax._offset+ax.d2p(args.x),y:ay._offset+ay.d2p(args.y)}}''',{'x':x,'y':y,'subplot':subplot})
    box=plot.bounding_box()
    page.mouse.move(box['x']+relative['x'],box['y']+relative['y'])
    output=frame.locator('[data-chart="'+name+'"] .cursor-readout')
    output.wait_for()
    frame.wait_for_function('(name)=>document.querySelector(`[data-chart="${name}"] .cursor-readout`).dataset.x!==undefined',arg=name)
    values=output.evaluate('(el)=>({x:el.dataset.x,y:el.dataset.y,text:el.textContent})')
    return values

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.chromium_path,headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':1600,'height':1100},accept_downloads=True)
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(args.url)
    page.get_by_role('button',name='Enter the prediction lab →').wait_for(timeout=90000)
    page.get_by_role('button',name='Enter the prediction lab →').click()
    page.get_by_role('button',name='Predict response',exact=True).wait_for()
    with page.expect_download() as download:
        page.get_by_role('button',name='Download populated feature example (CSV)',exact=True).click()
    download.value.save_as(OUT/'sample.csv')
    page.get_by_role('button',name='Predict response',exact=True).click()
    page.get_by_role('heading',name='Understand the IC50 scale',exact=True).wait_for(timeout=60000)
    frame=plot_frame(page)
    result=coordinates(page,frame,'ic50_concentration_scale',1,'Predicted')
    assert abs(float(result['x'])-1)<.03,result
    print('Predict log-axis cursor:',result,flush=True)
    page.get_by_test_id('stFileUploader').locator('input[type=file]').set_input_files(str(OUT/'sample.csv'))
    page.get_by_role('button',name='Download all predictions',exact=True).wait_for(timeout=60000)
    page.locator('.st-key-navigation').get_by_text('Explainability',exact=True).click()
    page.get_by_role('button',name='Calculate feature-level SHAP values for this pair',exact=True).wait_for(timeout=60000)
    page.get_by_role('button',name='Calculate feature-level SHAP values for this pair',exact=True).click()
    page.get_by_role('button',name='Download local feature SHAP values (.csv)',exact=True).wait_for(timeout=60000)
    page.locator('.st-key-navigation').get_by_text('Evaluation',exact=True).click()
    page.get_by_role('button',name='Prepare complete report and evidence bundle',exact=True).wait_for(timeout=120000)
    assert page.get_by_test_id('stMetric').count()==0
    assert page.get_by_test_id('stAlert').count()==0
    assert page.get_by_test_id('stJson').count()==0
    frame=plot_frame(page)
    names=frame.evaluate('window.pathdspCharts.map(spec=>spec.name)')
    assert names==['interactive_holdout_view','gdsc_agreement_density','ccle_agreement_density'],names
    result=coordinates(page,frame,'interactive_holdout_view',.5,.5)
    assert abs(float(result['x'])-.5)<.01 and abs(float(result['y'])-.5)<.01,result
    result=coordinates(page,frame,'gdsc_agreement_density',-1.5,-1.5)
    assert abs(float(result['x'])+1.5)<.03,result
    result=coordinates(page,frame,'ccle_agreement_density',0,0)
    assert abs(float(result['x']))<.03,result
    print('Evaluation has exactly the requested three interactive graphs',flush=True)
    page.get_by_text('Dark mode',exact=True).click()
    page.wait_for_function("Array.from(document.querySelectorAll('iframe')).some(f=>{try{return f.contentDocument.querySelector('.interactive-gallery.dark')&&f.contentWindow.pathdspPlotsReady}catch(e){return false}})",timeout=120000)
    page.get_by_role('button',name='Prepare complete report and evidence bundle',exact=True).wait_for()
    frame=plot_frame(page)
    coordinates(page,frame,'gdsc_agreement_density',-1.5,-1.5)
    page.screenshot(path=str(OUT/'evaluation-interactive-dark.png'))
    page.get_by_role('button',name='Prepare complete report and evidence bundle',exact=True).click()
    page.get_by_role('button',name='Download evaluation report (HTML)',exact=True).wait_for(timeout=120000)
    with page.expect_download() as download:
        page.get_by_role('button',name='Download evaluation report (HTML)',exact=True).click()
    download.value.save_as(OUT/'report.html')
    with page.expect_download() as download:
        page.get_by_role('button',name='Download complete evidence (ZIP)',exact=True).click()
    download.value.save_as(OUT/'evidence.zip')
    with zipfile.ZipFile(OUT/'evidence.zip') as archive:
        frames=[name for name in archive.namelist() if name.startswith('tables/workspace_')]
        assert any(name.endswith('_batch_predictions.csv') for name in frames),frames
        assert any(name.endswith('_feature_shap.csv') for name in frames),frames
        assert len(pd.read_csv(BytesIO(archive.read('tables/saved_holdout_predictions.csv'))))==7547
        inventory=pd.read_csv(BytesIO(archive.read('tables/evaluation_table_inventory.csv')))
        assert all('tables/'+name+'.csv' in archive.namelist() for name in inventory['Table'])
    offline=browser.new_page(viewport={'width':1400,'height':1100},accept_downloads=True)
    external=[]
    def route(request_route):
        if urlparse(request_route.request.url).hostname in ('127.0.0.1','localhost'):request_route.continue_()
        else:external.append(request_route.request.url);request_route.abort()
    offline.route('**/*',route)
    offline.goto(REPORT_URL,timeout=120000)
    offline.evaluate('window.pathdspPlotsReady')
    result=coordinates(offline,offline.main_frame,'ic50_concentration_scale',1,'Predicted')
    assert abs(float(result['x'])-1)<.03,result
    result=coordinates(offline,offline.main_frame,'saved_confusion_matrix','Below training median','Below training median')
    assert result['x']=='Below training median',result
    matrix=offline.locator('[data-chart="saved_correlation_heatmap"] .plot-area')
    first=matrix.evaluate('(gd)=>gd.layout.xaxis.categoryarray ? gd.layout.xaxis.categoryarray[0] : gd.data[0].x[0]')
    result=coordinates(offline,offline.main_frame,'saved_correlation_heatmap',first,first)
    assert result['x']==first,result
    assert not external,external
    assert offline.get_by_role('heading',name='Methods · scientific design',exact=True).count()==1
    assert offline.get_by_text('pip install -r requirements.txt',exact=True).count()==0
    offline.screenshot(path=str(OUT/'offline-coordinate-heatmap.png'))
    # The HTML itself contains complete CSVs, independent of the ZIP and preview length.
    link=offline.locator('a[download="saved_holdout_predictions.csv"]')
    csv=base64.b64decode(link.get_attribute('href').split('base64,',1)[1])
    assert len(pd.read_csv(BytesIO(csv)))==7547
    details=offline.locator('details[data-full-table="saved_holdout_predictions"]')
    details.locator('summary').click()
    details.get_by_role('button',name='Next rows',exact=True).click()
    assert details.locator('[data-table-status]').inner_text()=='Rows 26–50 of 7547'
    original=pd.read_csv(BytesIO(csv))
    assert details.locator('tbody tr').first.locator('td').first.inner_text()==str(original.iloc[25,0])

    assert not errors,errors
    mobile=browser.new_page(viewport={'width':390,'height':844})
    mobile.goto(args.url);mobile.get_by_role('button',name='Enter the prediction lab →').wait_for(timeout=90000)
    assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    mobile.locator('.st-key-navigation').get_by_text('Evaluation',exact=True).click()
    mobile.get_by_role('button',name='Prepare complete report and evidence bundle',exact=True).wait_for(timeout=120000)
    mobile_frame=plot_frame(mobile)
    mobile_result=coordinates(mobile,mobile_frame,'gdsc_agreement_density',-1.5,-1.5)
    assert abs(float(mobile_result['x'])+1.5)<.1,mobile_result
    assert mobile.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    mobile.screenshot(path=str(OUT/'mobile-evaluation-coordinates.png'))
    print('PASS: real numeric/log/category/subplot coordinates; app light/dark; offline report heatmap; full generated session data; no external requests; mobile',flush=True)
    browser.close()
server.shutdown()
server.server_close()
