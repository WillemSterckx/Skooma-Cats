NB. ============================================================
NB. mushroom_api.ijs
NB. A tiny HTTP API in J.
NB.   GET  /          -> HTML form
NB.   POST /predict   -> JSON { "class": 0|1 }
NB. Accepts application/x-www-form-urlencoded OR application/json
NB. ============================================================


load 'socket'
load 'strings'
load 'convert/json'

cocurrent 'mushroom'
coinsert 'jsocket'

NB. ------------------------------------------------------------
NB. Config
NB. ------------------------------------------------------------
PORT =: 8080
HOST =: '0.0.0.0'
CRLF =: 13 10 { a.
LF   =: 10 { a.

NB. ------------------------------------------------------------
NB. Feature schema (order is canonical — the model expects this)
NB. ------------------------------------------------------------
FEATURES =: ;: 'cap-diameter stem-height stem-width spore-print-color gill-color habitat season ring-type cap-shape stem-surface jumbled_noise_0 jumbled_noise_1'

NB. Numeric columns (for the future real model). Categoricals are
NB. expected to be one-hot / ordinal encoded downstream.
NUMERIC_FEATURES =: ;: 'cap-diameter stem-height stem-width'

CAT_FEATURES =: ;: 'spore-print-color gill-color habitat season ring-type cap-shape stem-surface jumbled_noise_0 jumbled_noise_1'
CLASS_LABELS =: 'e' ; 'p'

NB. ------------------------------------------------------------
NB. Model interface  (STUB — swap these two verbs for a real model)
NB. ------------------------------------------------------------
model_init =: 3 : 0
  smoutput 'mushroom_api: model_init (stub)'
  1
)

NB. predict =: 3 : 0
NB.   nums =. ". > each (NUMERIC_FEATURES i. FEATURES) { y
NB.   cats =. (CAT_FEATURES i. FEATURES) { y
NB.   x    =. (nums , encode_cats cats) , 1
NB.   p    =. 1 % 1 + ^ - (WEIGHTS +/ . * x)
NB.   (> (p > 0.5) { CLASS_LABELS)
NB. )

NB. predict :: boxed vector (length #FEATURES) -> character label
predict =: 3 : 0
  s =. ; ": each y
  h =. +/ a. i. s
  (> (2 | h) { CLASS_LABELS)
)

NB. ------------------------------------------------------------
NB. HTTP helpers
NB. ------------------------------------------------------------
find_crlfcrlf =: 3 : 0
  m =. (CRLF,CRLF) E. y
  i =. m i. 1
  if. i = # m do. _1 else. i end.
)

content_length =: 3 : 0
  h =. tolower y
  m =. 'content-length:' E. h
  i =. m i. 1
  if. i = # m do. 0 return. end.
  rest =. (i + 15) }. h
  e =. rest i. 13
  ". dlb e {. rest
)

recv_request =: 3 : 0
  sock =. y
  buf =. ''
  hdr_end =. _1
  clen =. 0
  whilst. 1 do.
    r     =. sdrecv sock ; 8192 ; 0
    chunk =. > 1 { r
    if. 0 = # chunk do. break. end.
    buf =. buf , chunk
    if. hdr_end = _1 do.
      hdr_end =. find_crlfcrlf buf
      if. hdr_end > _1 do.
        clen =. content_length (hdr_end + 4) {. buf
      end.
    end.
    if. (hdr_end > _1) *. (# buf) >: hdr_end + 4 + clen do. break. end.
  end.
  buf
)

parse_request =: 3 : 0
  buf =. y
  i =. find_crlfcrlf buf
  if. i = _1 do.
    hdrpart =. buf
    body =. ''
  else.
    hdrpart =. i {. buf
    body =. (i + 4) }. buf
  end.
  lines =. <;._2 hdrpart , CRLF
  if. 0 = # lines do. '' ; '' ; '' return. end.
  toks =. ' ' cut > 0 { lines
  if. 2 > # toks do. '' ; '' ; body return. end.
  method =. > 0 { toks
  path   =. > 1 { toks
  method ; path ; body
)

status_text =: 3 : 0
  select. y
  case. 200 do. 'OK'
  case. 204 do. 'No Content'
  case. 400 do. 'Bad Request'
  case. 404 do. 'Not Found'
  case. 405 do. 'Method Not Allowed'
  case. 500 do. 'Internal Server Error'
  case. do. 'Unknown'
  end.
)

build_response =: 3 : 0
  'status ct body' =. y
  h =. 'HTTP/1.1 ' , (": status) , ' ' , (status_text status) , CRLF
  h =. h , 'Content-Type: ' , ct , CRLF
  h =. h , 'Content-Length: ' , (": # body) , CRLF
  h =. h , 'Access-Control-Allow-Origin: *' , CRLF
  h =. h , 'Access-Control-Allow-Methods: GET, POST, OPTIONS' , CRLF
  h =. h , 'Access-Control-Allow-Headers: Content-Type' , CRLF
  h =. h , 'Connection: close' , CRLF
  h =. h , CRLF
  h , body
)

NB. ------------------------------------------------------------
NB. URL-encoded body parsing
NB. ------------------------------------------------------------
hexval =: 3 : 0
  v =. a. i. y
  if. (v >: 48) *. v <: 57  do. v - 48       return. end.
  if. (v >: 97) *. v <: 102 do. 10 + v - 97  return. end.
  if. (v >: 65) *. v <: 70  do. 10 + v - 65  return. end.
  _1
)

url_decode =: 3 : 0
  s =. y
  plus =. (s = '+') # i. # s
  s =. ' ' plus } s
  out =. ''
  i =. 0
  whilst. i < # s do.
    c =. i { s
    if. c = '%' do.
      if. (i + 2) <: # s do.
        h1 =. hexval (i + 1) { s
        h2 =. hexval (i + 2) { s
        if. (h1 >: 0) *. (h2 >: 0) do.
          out =. out , a. {~ 16 #. h1 , h2
          i =. i + 3
        else.
          out =. out , c
          i =. i + 1
        end.
      else.
        out =. out , c
        i =. i + 1
      end.
    else.
      out =. out , c
      i =. i + 1
    end.
  end.
  out
)

parse_urlencoded =: 3 : 0
  pairs =. <;._1 '&' , y
  ks =. 0 $ <''
  vs =. 0 $ <''
  for_p. pairs do.
    pair =. > p
    if. 0 = # pair do. continue. end.
    eq =. pair i. '='
    if. eq = # pair do.
      k =. url_decode pair
      v =. ''
    else.
      k =. url_decode eq {. pair
      v =. url_decode (eq + 1) }. pair
    end.
    ks =. ks , < k
    vs =. vs , < v
  end.
  ks ; vs
)

NB. ------------------------------------------------------------
NB. JSON body parsing (flat object only)
NB. ------------------------------------------------------------
parse_json_obj =: 3 : 0
  try.   obj =. dec_json y
  catch. _1 return. end.
  if. 2 = # $ obj do.
    if.     2 = {. $ obj do. (0 { obj)    ; (1 { obj)
    elseif. 2 = {: $ obj do. (0 {"1 obj) ; (1 {"1 obj)
    else.   _1
    end.
  else.
    _1
  end.
)

parse_body =: 3 : 0
  b =. y
  if. 0 = # b do. _1 return. end.
  if. '{' = {. b do. parse_json_obj b
  else.            parse_urlencoded b
  end.
)

NB. ------------------------------------------------------------
NB. Feature extraction — build a canonical-order boxed vector
NB. ------------------------------------------------------------
extract_features =: 3 : 0
  'ks vs' =. y
  ks =. , ks
  vs =. , vs
  if. -. *./ FEATURES e. ks do. _1 return. end.
  res    =. (# FEATURES) $ <''
  filled =. (# FEATURES) $ 0
  for_i. i. # ks do.
    nm =. > i { ks
    j  =. FEATURES i. nm
    if. j < # FEATURES do.
      res    =. (< (> i { vs)) j } res
      filled =. 1 j } filled
    end.
  end.
  if. 0 e. filled do. _1 return. end.
  res
)

NB. Scalar float check. Returns 1 if y parses as a single real number.
is_number =: 3 : 0
  r =. 0
  try.
    v =. ". y
    r =. (1 = # v) *. (v = v)   NB. scalar and not NaN
  catch.
    r =. 0
  end.
  r
)

NB. Given ('ks' ; 'vs') as produced by parse_body, verify every numeric
NB. feature is a parseable float. Returns 1/0.
validate_numeric =: 3 : 0
  'ks vs' =. y
  ks =. , ks
  vs =. , vs
  ok =. 1
  for_n. NUMERIC_FEATURES do.
    j =. ks i. > n
    if. j = # ks do. ok =. 0 break. end.
    if. -. is_number > j { vs do. ok =. 0 break. end.
  end.
  ok
)

NB. ------------------------------------------------------------
NB. Handlers
NB. ------------------------------------------------------------
handle_predict =: 3 : 0
  kv =. parse_body y
  if. kv -: _1 do.
    400 ; 'application/json' ; '{"error":"unparseable request body"}' return.
  end.

  if. -. validate_numeric kv do.
    400 ; 'application/json' ; '{"error":"numeric fields must be valid numbers: cap-diameter, stem-height, stem-width"}' return.
  end.

  feats =. extract_features kv
  if. feats -: _1 do.
    400 ; 'application/json' ; '{"error":"missing or empty feature fields"}' return.
  end.

  pred =. predict feats
  if. 0 = # , pred do.
    500 ; 'application/json' ; '{"error":"model returned no label"}' return.
  end.

  200 ; 'application/json' ; '{"class":"' , (; pred) , '"}'
)

form_page =: 3 : 0
  inputs =. ''
  for_f. FEATURES do.
    nm =. > f
    inputs =. inputs , '<label>' , nm , ' <input name="' , nm , '" /></label><br/>' , LF
  end.
  t =. ''
  t =. t , '<!DOCTYPE html>' , LF
  t =. t , '<html><head><meta charset="utf-8"><title>Mushroom Classifier</title></head><body>' , LF
  t =. t , '<h1>Mushroom Classifier</h1>' , LF
  t =. t , '<form id="f">' , LF , inputs
  t =. t , '<button type="submit">Predict</button>' , LF
  t =. t , '</form>' , LF
  t =. t , '<pre id="out"></pre>' , LF
  t =. t , '<script>' , LF
  t =. t , 'document.getElementById("f").addEventListener("submit", async (e) => {' , LF
  t =. t , '  e.preventDefault();' , LF
  t =. t , '  const fd = new FormData(e.target);' , LF
  t =. t , '  const body = new URLSearchParams(fd).toString();' , LF
  t =. t , '  const r = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded"}, body:body});' , LF
  t =. t , '  const j = await r.json();' , LF
  t =. t , '  document.getElementById("out").textContent = JSON.stringify(j, null, 2);' , LF
  t =. t , '});' , LF
  t =. t , '</script>' , LF
  t =. t , '</body></html>'
  t
)

route =: 3 : 0
  'method path body' =. y
  select. method
  case. 'GET' do.
    if. path -: '/' do.
      200 ; 'text/html; charset=utf-8' ; form_page ''
    else.
      404 ; 'text/plain' ; 'Not Found'
    end.
  case. 'POST' do.
    if. path -: '/predict' do.
      handle_predict body
    else.
      404 ; 'text/plain' ; 'Not Found'
    end.
  case. 'OPTIONS' do.
    204 ; 'text/plain' ; ''
  case. do.
    405 ; 'text/plain' ; 'Method Not Allowed'
  end.
)

handle_conn =: 3 : 0
  sock =. {. , y
  smoutput 'sock type:  ' , datatype sock
  smoutput 'sock shape: ' , ": $ sock
  smoutput 'sock value: ' , ": ; , sock
  req =. recv_request sock
  if. 0 = # req do.
    sdclose sock
    return.
  end.
  'method path body' =. parse_request req
  resp =. route method ; path ; body
  r =. build_response resp
  rc =. r sdsend sock ; 0
  if. rc do. smoutput 'send failed: ' , ": rc end.
  sdclose sock
)

NB. ------------------------------------------------------------
NB. Main loop
NB. ------------------------------------------------------------
run =: 3 : 0
  model_init ''
  r =. sdsocket ''
  s =. > 1 { r
  if. _1 = s do.
    smoutput 'ERROR: could not create socket' return.
  end.
  rc =. sdbind s ; AF_INET ; HOST ; PORT
  if. rc do.
    smoutput 'ERROR: bind failed, code ' , ": rc , ' (10048 = port in use)'
    sdclose s return.
  end.
  rc =. sdlisten s
  if. rc do.
    smoutput 'ERROR: listen failed, code ' , ": rc
    sdclose s return.
  end.
  smoutput 'mushroom_api listening on ' , HOST , ':' , (": PORT)
  whilst. 1 do.
    r   =. sdaccept s
    rc  =. > 0 { r
    cli =. {. , > 1 { r
    if. rc = 0 do.
      try.
        handle_conn cli
      catch.
        smoutput 'handler error: ' , (13!:12 '')
        try. sdclose cli catch. end.
      end.
    else.
      smoutput 'accept failed: ' , ": rc
      6!:3 (0.5)      NB. sleep half a second, don't spin
    end.
  end.
)