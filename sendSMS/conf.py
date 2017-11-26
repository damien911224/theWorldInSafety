import base64

appid = 'twis'
apikey = 'api-key'
address = 'api.bluehouselab.com'

sender = 'phone-number'
receivers = ['phone-number', ]
content = u'Hi ru'  # url

credential = "Basic "+base64.encodestring(appid+':'+apikey).strip()
headers = {
  "Content-type": "application/json;charset=utf-8",
  "Authorization": credential,
}