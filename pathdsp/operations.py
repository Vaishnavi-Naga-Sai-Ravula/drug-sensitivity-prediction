"""Bounded per-process API controls. Shared ingress limits still required for replicas."""
from collections import OrderedDict
import math
import os
import secrets
import threading
import time
import uuid
from starlette.responses import JSONResponse

class ApiControls:
    def __init__(self):
        self.lock=threading.Lock()
        self.buckets=OrderedDict()

    def allow(self, identity, limit):
        now=time.monotonic()
        with self.lock:
            tokens, previous=self.buckets.pop(identity,(float(limit),now))
            tokens=min(limit,tokens+(now-previous)*limit/60)
            allowed=tokens>=1
            self.buckets[identity]=(tokens-1 if allowed else tokens,now)
            if len(self.buckets)>1024:self.buckets.popitem(last=False)
            return allowed,max(1,math.ceil((1-tokens)*60/limit))

    async def middleware(self,request,call_next):
        request_id=uuid.uuid4().hex
        start=time.perf_counter()
        key=os.environ.get('PATHDSP_API_KEY','')
        if key and request.url.path not in ('/health/live','/health/ready'):
            supplied=request.headers.get('authorization','')
            if not secrets.compare_digest(supplied.encode(),('Bearer '+key).encode()):
                return JSONResponse({'detail':'Authentication required','request_id':request_id},status_code=401,headers={'WWW-Authenticate':'Bearer'})
        try:limit=int(os.environ.get('PATHDSP_REQUESTS_PER_MINUTE','120'))
        except ValueError:limit=120
        limit=max(1,min(limit,10000))
        if request.url.path.startswith('/v1/'):
            allowed,retry=self.allow(request.client.host if request.client else 'local',limit)
            if not allowed:return JSONResponse({'detail':'Request limit reached','request_id':request_id},status_code=429,headers={'Retry-After':str(retry)})
        response=await call_next(request)
        response.headers['X-Request-ID']=request_id
        response.headers['Server-Timing']=f'app;dur={(time.perf_counter()-start)*1000:.2f}'
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        return response
