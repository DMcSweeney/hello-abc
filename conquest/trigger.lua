local socket = require("socket")
client = socket.connect("backend", 5001)
if client then
    client:send("GET /api/conquest/handle_trigger?series_uid="..command_line.." HTTP/1.1\r\n\r\n")
    client:close()
end