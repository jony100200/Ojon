//! Single-model, loopback-only evaluation sessions for deterministic task suites.
//! Model weights stay loaded across prompts; the process is killed/reaped on Drop.
//! Does not reuse server state across different models or separate jobs.
use serde_json::{json,Value};
use std::io::{self,Read,Write,ErrorKind};
use std::net::{TcpListener,TcpStream,SocketAddr};
use std::path::Path;
use std::process::{Child,Command,Stdio};
use std::thread;
use std::time::{Duration,Instant};

fn bad(s:impl Into<String>)->io::Error{io::Error::new(ErrorKind::InvalidData,s.into())}
const MAX_HTTP_BYTES:usize=2*1024*1024;
const READY_TIMEOUT:Duration=Duration::from_secs(180);
const ANSWER_TIMEOUT:Duration=Duration::from_secs(600);

/// Decode HTTP/1.1 payloads with both Content-Length and chunked transport.
/// Never accept unbounded/malformed bodies as task evidence.
fn body_from_http(response:&[u8])->io::Result<(u16,Vec<u8>)>{
    let boundary=response.windows(4).position(|w|w==b"\r\n\r\n")
        .ok_or_else(||bad("model server returned invalid HTTP"))?;
    let (header,content)=response.split_at(boundary+4);
    let header=std::str::from_utf8(header).map_err(|_|bad("non-UTF8 HTTP headers"))?;
    let status=header.lines().next().and_then(|line|line.split_whitespace().nth(1))
        .and_then(|n|n.parse::<u16>().ok()).ok_or_else(||bad("no HTTP status"))?;
    let chunked=header.lines().any(|line|{
        line.split_once(':').is_some_and(|(k,v)|
            k.eq_ignore_ascii_case("transfer-encoding")&&v.to_ascii_lowercase().contains("chunked"))
    });
    let body=if chunked{
        let mut decoded=Vec::new();
        let mut rest=content;
        loop{
            let end=rest.windows(2).position(|w|w==b"\r\n")
                .ok_or_else(||bad("missing chunk delimiter"))?;
            let code=std::str::from_utf8(&rest[..end]).map_err(|_|bad("invalid chunk header"))?;
            let size=usize::from_str_radix(code.split(';').next().unwrap_or(""),16)
                .map_err(|_|bad("invalid chunk size"))?;
            rest=&rest[end+2..];
            if size==0{break;}
            if size>MAX_HTTP_BYTES.saturating_sub(decoded.len())||rest.len()<size+2
                ||&rest[size..size+2]!=b"\r\n"{
                return Err(bad("truncated or oversized chunk"));
            }
            decoded.extend_from_slice(&rest[..size]);
            rest=&rest[size+2..];
        }
        decoded
    }else{
        let length=header.lines().find_map(|line|{
            line.split_once(':').and_then(|(k,v)|
                k.eq_ignore_ascii_case("content-length").then(||v.trim().parse::<usize>().ok())).flatten()
        });
        if let Some(len)=length{
            if len>content.len(){return Err(bad("truncated HTTP content"));}
            content[..len].to_vec()
        }else{content.to_vec()}
    };
    if body.len()>MAX_HTTP_BYTES{return Err(bad("model server HTTP body too large"));}
    Ok((status,body))
}
fn call(addr:SocketAddr,path:&str,body:Option<&[u8]>,timeout:Duration)->io::Result<(u16,Value)>{
    let mut stream=TcpStream::connect_timeout(&addr,Duration::from_millis(750))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let method=if body.is_some(){"POST"}else{"GET"};
    let data=body.unwrap_or(&[]);
    let header=format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",data.len());
    stream.write_all(header.as_bytes())?;
    if !data.is_empty(){stream.write_all(data)?;}
    let mut response=Vec::new();
    // Reading to EOF with "Connection: close" avoids any dependency on an
    // HTTP client crate; the bound applies to headers AND response content.
    let mut block=[0u8;8192];
    loop{
        let n=stream.read(&mut block)?;
        if n==0{break;}
        if response.len().saturating_add(n)>MAX_HTTP_BYTES{
            return Err(bad("model server response exceeded 2 MiB"));
        }
        response.extend_from_slice(&block[..n]);
    }
    let (status,body)=body_from_http(&response)?;
    let value=serde_json::from_slice(&body).map_err(|_|bad("model server returned invalid JSON"))?;
    Ok((status,value))
}
pub struct EvalSession{
    child:Child,
    address:SocketAddr,
}
impl Drop for EvalSession{
    fn drop(&mut self){
        let _=self.child.kill();
        let _=self.child.wait();
    }
}
impl EvalSession{
    pub fn launch(server:&Path,model:&Path,gpu_layers:i32,device:Option<&str>)->io::Result<Self>{
        if !server.is_file(){return Err(io::Error::new(ErrorKind::NotFound,
            "configured llama-server binary does not exist"));}
        let listener=TcpListener::bind(("127.0.0.1",0))?;
        let address=listener.local_addr()?;
        drop(listener);
        let mut command=Command::new(server);
        command.arg("-m").arg(model)
            .arg("--host").arg("127.0.0.1")
            .arg("--port").arg(address.port().to_string())
            .arg("-ngl").arg(gpu_layers.to_string())
            .arg("--parallel").arg("1");
        if let Some(device)=device{command.arg("-dev").arg(device);}
        // Discard the native server's logging output (including large startup
        // banners); this process does not inherit access to Ojon's UI stdin.
        let child=command.stdin(Stdio::null()).stdout(Stdio::null())
            .stderr(Stdio::null()).spawn()?;
        let mut session=Self{child,address};
        let started=Instant::now();
        loop{
            if let Some(status)=session.child.try_wait()?{
                return Err(bad(format!("llama-server exited during model startup: {status}")));
            }
            if let Ok((200,response))=call(address,"/health",None,Duration::from_secs(2)){
                if response["status"]=="ok" {return Ok(session);}
            }
            if started.elapsed()>=READY_TIMEOUT{
                return Err(io::Error::new(ErrorKind::TimedOut,
                    "local llama-server did not finish loading model within 180 seconds"));
            }
            thread::sleep(Duration::from_millis(300));
        }
    }
    pub fn completion(&mut self,prompt:&str,tokens:u32)->io::Result<String>{
        if let Some(status)=self.child.try_wait()?{
            return Err(bad(format!("llama-server exited before task completion: {status}")));
        }
        let request=json!({
            "prompt":prompt,"n_predict":tokens,"temperature":0.0,
            "seed":42,"stream":false,"cache_prompt":false,
            "n_keep":0
        });
        let encoded=serde_json::to_vec(&request).map_err(io::Error::other)?;
        let (status,response)=call(self.address,"/completion",Some(&encoded),ANSWER_TIMEOUT)?;
        if status!=200{return Err(bad(format!("llama-server completion HTTP {status}")));}
        if response.get("error").is_some(){return Err(bad("llama-server reported completion error"));}
        if response["truncated"]==true{
            return Err(bad("llama-server truncated task context; cannot grade this answer"));
        }
        response["content"].as_str().map(str::to_owned)
            .ok_or_else(||bad("missing completion text"))
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn parser_accepts_chunked_and_content_length(){
        let a=b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}";
        assert_eq!(body_from_http(a).unwrap().0,200);
        let b=b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nF\r\n{\"status\":\"ok\"}\r\n0\r\n\r\n";
        assert_eq!(body_from_http(b).unwrap().1,b"{\"status\":\"ok\"}");
        assert!(body_from_http(b"wrong").is_err());
    }
}
