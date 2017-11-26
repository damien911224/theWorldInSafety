import base64

appid = 'twis'
apikey = 'cb4eb398d21a11e794990cc47a1fcfae'
address = 'api.bluehouselab.com'

sender = '01091261777'
receivers = ['01091261777', '01031345174', ]
content = u'Hi ru'  # url

credential = "Basic "+base64.encodestring(appid+':'+apikey).strip()
headers = {
  "Content-type": "application/json;charset=utf-8",
  "Authorization": credential,
}