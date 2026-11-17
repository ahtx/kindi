#!/usr/bin/env python3
import socket, select, sys, threading

def handle(client):
    try:
        req = b''
        while b'\r\n\r\n' not in req:
            chunk = client.recv(4096)
            if not chunk: return
            req += chunk
        first = req.split(b'\r\n',1)[0].decode('iso-8859-1')
        method, target, _ = first.split(' ',2)
        headers = req.split(b'\r\n\r\n',1)[0]
        if method.upper() == 'CONNECT':
            host, port = (target.split(':',1)+['443'])[:2]
            port = int(port)
            remote = socket.create_connection((host, port))
            client.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            relay(client, remote)
        else:
            # absolute-form URL
            url = target
            if url.startswith('http://'):
                url = url[7:]
            hostport, _, path = url.partition('/')
            path = '/' + path
            if ':' in hostport:
                host, port = hostport.split(':',1); port = int(port)
            else:
                host, port = hostport, 80
            remote = socket.create_connection((host, port))
            # rewrite request line to origin-form
            rest = req.split(b'\r\n',1)[1] if b'\r\n' in req else b''
            remote.sendall(('GET %s HTTP/1.1\r\n' % path).encode() + rest if method.upper()=='GET' else (first.split(' ',1)[0]+' '+path+' HTTP/1.1\r\n').encode()+rest)
            relay(client, remote)
    except Exception:
        pass
    finally:
        try: client.close()
        except: pass

def relay(a,b):
    sockets = [a,b]
    try:
        while True:
            r,_,_ = select.select(sockets, [], [], 60)
            if not r: break
            for s in r:
                data = s.recv(65536)
                if not data: return
                (b if s is a else a).sendall(data)
    except Exception:
        pass
    finally:
        for s in sockets:
            try: s.close()
            except: pass

def main():
    port = int(sys.argv[1]) if len(sys.argv)>1 else 8888
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(128)
    print('PROXY_LISTEN', port, flush=True)
    while True:
        c,_ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()

if __name__ == '__main__':
    main()
