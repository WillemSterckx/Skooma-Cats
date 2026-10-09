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
FEATURES =: ' ' cut 'cap-diameter stem-height stem-width spore-print-color gill-color habitat season ring-type cap-shape stem-surface jumbled_noise_0 jumbled_noise_1'

NB. Numeric columns (for the future real model). Categoricals are
NB. expected to be one-hot / ordinal encoded downstream.
NUMERIC_FEATURES =: ' ' cut 'cap-diameter stem-height stem-width'

CAT_FEATURES =: ' ' cut 'spore-print-color gill-color habitat season ring-type cap-shape stem-surface jumbled_noise_0 jumbled_noise_1'
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
  if. i = # m do. return. end.
  rest =. (i + 15) }. h
  e1 =. rest i. 13 { a.
  e2 =. rest i. 10 { a.
  e =. e1 <. e2
  val =. e {. rest
  val =. val -. 13 { a.
  val =. val -. 10 { a.
  ". dlb val
)

recv_request =: 3 : 0
  sock =. y
  buffer =. ''
  header_end =. _1
  content_len =. 0
  whilst. 1 do.
    r     =. sdrecv sock ; 8192 ; 0
    data =. > 1 { r
    if. 0 = # data do. break. end.
    buffer =. buffer , data
    if. header_end = _1 do.
      header_end =. find_crlfcrlf buffer
      if. header_end > _1 do.
        content_len =. content_length (header_end + 4) {. buffer
      end.
    end.
    if. (header_end > _1) *. (# buffer) >: header_end + 4 + content_len do. break. end.
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
  plus_positions =. (s = '+') # i. # s
  s =. ' ' plus_positions } s
  output =. ''
  i =. 0
  whilst. i < # s do.
    c =. i { s
    if. c = '%' do.
      if. (i + 2) <: # s do.
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
NB. Feature extraction — build a canonical-order boxed vector
NB. ------------------------------------------------------------
extract_features =: 3 : 0
  'keys values' =. y
  if. -. *./ FEATURES e. keys do. _1 return. end.
  result =. (# FEATURES) $ <''
  for_i. i. # keys do.
    j =. FEATURES i. i { keys
    if. j < # FEATURES do. result =. (i { values) j } result end.
  end.
  result
)

NB. Scalar float check. Returns 1 if y parses as a single real number.
is_number =: 3 : 0
  v =. _. ". y          NB. Stops code injection
  (1 = # v) *. v = v
)

NB. Given ('ks' ; 'vs') as produced by , verify every numeric
NB. feature is a parseable float. Returns 1/0.
validate_numeric =: 3 : 0
  'keys values' =. y
  ok =. 1
  for_n. NUMERIC_FEATURES do.
    field_name =. > n
    j =. keys i. < field_name
    if. j = # keys do. ok =. 0 break. end.
    if. -. is_number > j { values do. ok =. 0 break. end.
  end.
  ok
)

NB. ------------------------------------------------------------
NB. Handlers
NB. ------------------------------------------------------------
handle_predict =: 3 : 0
  kvpair =. parse_body y
  smoutput 'KV type: ' , datatype kvpair
  smoutput 'KV shape: ' , ": $ kvpair
  smoutput 'KV length: ' , ": # , kvpair
  smoutput 'KV value: ' , ": kvpair
  if. kvpair -: _1 do.
    400 ; 'application/json' ; '{"error":"unparseable request body"}' return.
  end.

  if. -. validate_numeric kvpair do.
    400 ; 'application/json' ; '{"error":"numeric fields must be valid numbers: cap-diameter, stem-height, stem-width"}' return.
  end.

  features =. extract_features kvpair
  if. features -: _1 do.
    400 ; 'application/json' ; '{"error":"missing or empty feature fields"}' return.
  end.

  prediction =. predict features
  if. 0 = # , prediction do.
    500 ; 'application/json' ; '{"error":"model returned no label"}' return.
  end.

  200 ; 'application/json' ; '{"class":"' , (; prediction) , '"}'
)

form_page =: 3 : 0
  inputs =. ''
  for_f. FEATURES do.
    field_name =. > f
    inputs =. inputs , '<label>' , field_name , ' <input name="' , field_name , '" /></label><br/>' , LF
  end.
  html =. ''
  html =. html , '<!DOCTYPE html>' , LF
  html =. html , '<html><head><meta charset="utf-8"><title>Mushroom Classifier</title></head><body>' , LF
  html =. html , '<h1>Mushroom Classifier</h1>' , LF
  html =. html , '<form id="f">' , LF , inputs
  html =. html , '<button type="submit">Predict</button>' , LF
  html =. html , '</form>' , LF
  html =. html , '<pre id="out"></pre>' , LF
  html =. html , '<script>' , LF
  html =. html , 'document.getElementById("f").addEventListener("submit", async (e) => {' , LF
  html =. html , '  e.preventDefault();' , LF
  html =. html , '  const fd = new FormData(e.target);' , LF
  html =. html , '  const body = new URLSearchParams(fd).toString();' , LF
  html =. html , '  const r = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded"}, body:body});' , LF
  html =. html , '  const j = await r.json();' , LF
  html =. html , '  document.getElementById("out").textContent = JSON.stringify(j, null, 2);' , LF
  html =. html , '});' , LF
  html =. html , '</script>' , LF
  html =. html , '</body></html>'
  html
)

route =: 3 : 0
  'method path body' =. y
  smoutput 'ROUTE method=[' , method , '] path=[' , path , ']'
  select. method
  case. 'GET' do.
    smoutput 'GET branch, path bytes: ' , ": a. i. path
    smoutput 'literal bytes: ' , ": a. i. '/'
    smoutput 'comparison: ' , ": path -: 47 { a.
    smoutput 'path shape: ' , ": $ path
    if. path -: , 47 { a. do.
      200 ; 'text/html ; charset=utf-8' ; form_page ''
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
      6!:3 (0.5)      NB. sleep half a second, don't spin
    end.
  end.
)