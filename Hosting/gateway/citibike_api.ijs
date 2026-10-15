NB. ============================================================
NB. citibike_api.ijs
NB. Thin HTTP gateway in J. Forwards to the Python model service.
NB.   GET  /                -> HTML form
NB.   GET  /stations        -> proxied from model service
NB.   GET  /health          -> gateway liveness
NB.   POST /predict         -> JSON from model service
NB. Accepts application/x-www-form-urlencoded OR application/json
NB. ============================================================

load 'socket'
load 'strings'
load 'convert/json'

cocurrent 'citibike'
coinsert 'jsocket'

NB. ------------------------------------------------------------
NB. Config
NB. ------------------------------------------------------------
NB. x = default, y = environment variable name
env_or =: 4 : 0
  try. v =. 2!:5 y catch. v =. '' end.
  if. 0 = # v do. x else. v end.
)

PORT           =: 8080
HOST           =: '0.0.0.0'
CRLF           =: 13 10 { a.
LF             =: 10 { a.
CR             =: 13 { a.
MAX_BODY       =: 8192
MODEL_URL      =: 'http://model:8000' env_or 'MODEL_URL'
ALLOWED_ORIGIN =: '' env_or 'ALLOWED_ORIGIN'
MODEL_TMP      =: '/tmp/citibike_req.json'

NB. Inputs the form/API must supply (all sent to the model as strings).
FEATURES =: ' ' cut 'start_station_id member_casual started_at ended_at'

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
  e =. (rest i. CR) <. rest i. LF
  val =. ". dlb e {. rest
  if. 1 ~: # val do. 0 else. {. val end.
)

recv_request =: 3 : 0
  sock =. y
  buffer =. ''
  header_end =. _1
  content_len =. 0
  whilst. 1 do.
    r    =. sdrecv sock ; 8192 ; 0
    data =. > 1 { r
    if. 0 = # data do. break. end.
    buffer =. buffer , data
    if. header_end = _1 do.
      header_end =. find_crlfcrlf buffer
      if. header_end > _1 do.
        content_len =. content_length (header_end + 4) {. buffer
        if. content_len > MAX_BODY do. '' return. end.
      end.
    end.
    if. (header_end > _1) *. (# buffer) >: header_end + 4 + content_len do. break. end.
    if. (# buffer) > 16384 do. '' return. end.
  end.
  buffer
)

parse_request =: 3 : 0
  buffer =. y
  i =. find_crlfcrlf buffer
  if. i = _1 do.
    header_section =. buffer
    body =. ''
  else.
    header_section =. i {. buffer
    body =. (i + 4) }. buffer
  end.
  lines =. <;._2 header_section , CRLF
  if. 0 = # lines do. '' ; '' ; '' return. end.
  tokens =. ' ' cut > 0 { lines
  if. 2 > # tokens do. '' ; '' ; body return. end.
  method =. > 0 { tokens
  path   =. > 1 { tokens
  method ; path ; body
)

status_text =: 3 : 0
  select. y
  case. 200 do. 'OK'
  case. 204 do. 'No Content'
  case. 400 do. 'Bad Request'
  case. 404 do. 'Not Found'
  case. 405 do. 'Method Not Allowed'
  case. 422 do. 'Unprocessable Entity'
  case. 500 do. 'Internal Server Error'
  case. 502 do. 'Bad Gateway'
  case. do. 'Unknown'
  end.
)

build_response =: 3 : 0
  'status ct body' =. y
  h =. 'HTTP/1.1 ' , (": status) , ' ' , (status_text status) , CRLF
  h =. h , 'Content-Type: ' , ct , CRLF
  h =. h , 'Content-Length: ' , (": # body) , CRLF
  if. 0 < # ALLOWED_ORIGIN do.
    h =. h , 'Access-Control-Allow-Origin: ' , ALLOWED_ORIGIN , CRLF
    h =. h , 'Access-Control-Allow-Methods: GET, POST, OPTIONS' , CRLF
    h =. h , 'Access-Control-Allow-Headers: Content-Type, ngrok-skip-browser-warning' , CRLF
  end.
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
  plus_positions =. (s = '+') # i. # s
  s =. ' ' plus_positions } s
  output =. ''
  i =. 0
  whilst. i < # s do.
    c =. i { s
    if. c = '%' do.
      if. (i + 2) < # s do.
        h1 =. hexval (i + 1) { s
        h2 =. hexval (i + 2) { s
        if. (h1 >: 0) *. (h2 >: 0) do.
          output =. output , a. {~ 16 #. h1 , h2
          i =. i + 3
        else.
          output =. output , c
          i =. i + 1
        end.
      else.
        output =. output , c
        i =. i + 1
      end.
    else.
      output =. output , c
      i =. i + 1
    end.
  end.
  output
)

parse_urlencoded =: 3 : 0
  pairs =. <;._1 '&' , y
  keys =. 0 $ <''
  values =. 0 $ <''
  for_p. pairs do.
    pair =. > p
    if. 0 = # pair do. continue. end.
    equals_pos =. pair i. '='
    if. equals_pos = # pair do.
      k =. url_decode pair
      v =. ''
    else.
      k =. url_decode equals_pos {. pair
      v =. url_decode (equals_pos + 1) }. pair
    end.
    keys =. keys , < k
    values =. values , < v
  end.
  (< keys) , < values
)

NB. ------------------------------------------------------------
NB. JSON body parsing (flat object only)
NB. ------------------------------------------------------------
parse_json_obj =: 3 : 0
  try.   parsed =. dec_json y
  catch. _1 return. end.
  if. 2 = # $ parsed do.
    if.     2 = {. $ parsed do. (0 { parsed)    ; (1 { parsed)
    elseif. 2 = {: $ parsed do. (0 {"1 parsed) ; (1 {"1 parsed)
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
NB. Feature extraction: canonical-order boxed vector, or _1
NB. ------------------------------------------------------------
extract_features =: 3 : 0
  'keys values' =. y
  if. -. *./ FEATURES e. keys do. _1 return. end.
  result =. (# FEATURES) $ <''
  for_i. i. # keys do.
    j =. FEATURES i. i { keys
    if. j < # FEATURES do. result =. (i { values) j } result end.
  end.
  if. +./ 0 = > # each result do. _1 return. end.
  result
)

NB. ------------------------------------------------------------
NB. Talking to the Python model service (via curl, fixed command)
NB. ------------------------------------------------------------
json_escape =: 3 : 0
  r =. ''
  for_ch. y do.
    if.     ch = '"'         do. r =. r , '\"'
    elseif. ch = '\'         do. r =. r , '\\'
    elseif. (a. i. ch) < 32  do. r =. r , ' '
    else.                        r =. r , ch
    end.
  end.
  r
)

NB. boxed vector in FEATURES order -> JSON object of strings
build_payload =: 3 : 0
  items =. 0 $ <''
  for_i. i. # FEATURES do.
    k =. json_escape > i { FEATURES
    v =. json_escape > i { y
    items =. items , < '"' , k , '":"' , v , '"'
  end.
  '{' , (}. ; ',' ,each items) , '}'
)

NB. y = method ; path ; payload   ->   status code ; body   (code 0 = unreachable)
call_model =: 3 : 0
  'method path payload' =. y
  if. method -: 'POST' do.
    payload 1!:2 < MODEL_TMP
    cmd =. 'curl -s -m 5 -X POST -H "Content-Type: application/json" --data-binary @' , MODEL_TMP , ' -w "\n%{http_code}" ' , MODEL_URL , path
  else.
    cmd =. 'curl -s -m 5 -w "\n%{http_code}" ' , MODEL_URL , path
  end.
  try.
    out =. 2!:0 cmd
  catch.
    0 ; '' return.
  end.
  out =. out -. CR
  p   =. out i: LF
  if. p = # out do. 0 ; '' return. end.
  (". (p + 1) }. out) ; p {. out
)

proxy_get =: 3 : 0
  'code body' =. call_model 'GET' ; y ; ''
  if. code = 0 do.
    502 ; 'application/json' ; '{"error":"model service unavailable"}' return.
  end.
  code ; 'application/json' ; body
)

NB. ------------------------------------------------------------
NB. Handlers
NB. ------------------------------------------------------------
handle_predict =: 3 : 0
  kvpair =. parse_body y
  if. kvpair -: _1 do.
    400 ; 'application/json' ; '{"error":"unparseable request body"}' return.
  end.

  features =. extract_features kvpair
  if. features -: _1 do.
    400 ; 'application/json' ; '{"error":"missing or empty fields: start_station_id, member_casual, started_at, ended_at"}' return.
  end.

  'code body' =. call_model 'POST' ; '/predict' ; build_payload features
  if. code = 0 do.
    502 ; 'application/json' ; '{"error":"model service unavailable"}' return.
  end.
  code ; 'application/json' ; body
)

form_page =: 3 : 0
  html =. ''
  html =. html , '<!DOCTYPE html>' , LF
  html =. html , '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Citi Bike e-bike predictor</title></head><body>' , LF
  html =. html , '<h1>Citi Bike: electric bike or not?</h1>' , LF
  html =. html , '<form id="f">' , LF
  html =. html , '<label>Start station <select name="start_station_id" id="st" required></select></label><br/>' , LF
  html =. html , '<label>Rider type <select name="member_casual" required><option value="member">member</option><option value="casual">casual</option></select></label><br/>' , LF
  html =. html , '<label>Start time (NYC) <input type="datetime-local" name="started_at" required /></label><br/>' , LF
  html =. html , '<label>End time (NYC) <input type="datetime-local" name="ended_at" required /></label><br/>' , LF
  html =. html , '<button type="submit">Predict</button>' , LF
  html =. html , '</form>' , LF
  html =. html , '<pre id="out"></pre>' , LF
  html =. html , '<script>' , LF
  html =. html , 'async function fill(id, url, val, txt) {' , LF
  html =. html , '  const items = await (await fetch(url)).json();' , LF
  html =. html , '  const el = document.getElementById(id);' , LF
  html =. html , '  for (const x of items) { const o = document.createElement("option"); o.value = val(x); o.textContent = txt(x); el.appendChild(o); }' , LF
  html =. html , '}' , LF
  html =. html , 'fill("st", "/stations", x => x.id, x => x.name);' , LF
  html =. html , 'document.getElementById("f").addEventListener("submit", async (e) => {' , LF
  html =. html , '  e.preventDefault();' , LF
  html =. html , '  const body = new URLSearchParams(new FormData(e.target)).toString();' , LF
  html =. html , '  const r = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded"}, body:body});' , LF
  html =. html , '  document.getElementById("out").textContent = JSON.stringify(await r.json(), null, 2);' , LF
  html =. html , '});' , LF
  html =. html , '</script>' , LF
  html =. html , '</body></html>'
  html
)

route =: 3 : 0
  'method path body' =. y
  smoutput 'ROUTE ' , method , ' ' , path
  select. method
  case. 'GET' do.
    if.     path -: , '/'              do. 200 ; 'text/html; charset=utf-8' ; form_page ''
    elseif. path -: '/stations'        do. proxy_get '/stations'
    elseif. path -: '/health'          do. 200 ; 'application/json' ; '{"ok":true}'
    else.   404 ; 'text/plain' ; 'Not Found'
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
  request =. recv_request sock
  if. 0 = # request do.
    sdclose sock
    return.
  end.
  'method path body' =. parse_request request
  response =. route method ; path ; body
  r =. build_response response
  send_result =. r sdsend sock ; 0
  send_rc =. > 0 { send_result
  if. send_rc do. smoutput 'send failed: ' , ": send_rc end.
  sdclose sock
)

NB. ------------------------------------------------------------
NB. Main loop
NB. ------------------------------------------------------------
run =: 3 : 0
  r =. sdsocket ''
  s =. > 1 { r
  if. _1 = s do.
    smoutput 'ERROR: could not create socket' return.
  end.
  rc =. sdbind s ; AF_INET ; HOST ; PORT
  if. rc do.
    smoutput 'ERROR: bind failed, code ' , ": rc
    sdclose s return.
  end.
  rc =. sdlisten s
  if. rc do.
    smoutput 'ERROR: listen failed, code ' , ": rc
    sdclose s return.
  end.
  smoutput 'citibike gateway listening on ' , HOST , ':' , (": PORT)
  whilst. 1 do.
    r   =. sdaccept s
    rc  =. > 0 { r
    client_sock =. {. , > 1 { r
    if. rc = 0 do.
      try.
        handle_conn client_sock
      catch.
        smoutput 'handler error: ' , (13!:12 '')
        try. sdclose client_sock catch. end.
      end.
    else.
      smoutput 'accept failed: ' , ": rc
      6!:3 (0.5)
    end.
  end.
)
