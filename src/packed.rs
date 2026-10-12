//! Self-contained Rust weight-only quantization. No RebelUI, Python or CUDA.
//! OJONQ001 is an Ojon-owned packed container, NOT a GPU inference format.
//! Unpack restores ordinary safetensors without the original checkpoint.
//! We never claim its reconstruction error equals end-to-end model quality.

use serde_json::{json,Map,Value};
use std::fs::{self,File,OpenOptions};
use std::io::{self,Read,Seek,SeekFrom,Write};
use std::path::{Path,PathBuf};
use std::time::{SystemTime,UNIX_EPOCH};

const MAGIC:&[u8;8]=b"OJONQ001";
const HEADER_MAX:u64=32*1024*1024;
const ROW_MAX:usize=16*1024*1024;
fn bad(msg:impl Into<String>)->io::Error{io::Error::new(io::ErrorKind::InvalidData,msg.into())}
fn invalid(msg:impl Into<String>)->io::Error{io::Error::new(io::ErrorKind::InvalidInput,msg.into())}
fn width(name:&str)->Option<usize>{match name{
    "F16"|"BF16"|"I16"|"U16"=>Some(2),
    "F32"|"I32"|"U32"=>Some(4),
    "F64"|"I64"|"U64"=>Some(8),
    "I8"|"U8"|"BOOL"=>Some(1),
    _=>None
}}
fn checked_len(shape:&[u64],elem:usize)->io::Result<u64>{
    shape.iter().try_fold(elem as u64,|n,x|n.checked_mul(*x).ok_or_else(||bad("tensor size overflow")))
}
#[derive(Clone,Debug)]
struct Tensor {name:String,dtype:String,shape:Vec<u64>,start:u64,end:u64}
#[derive(Debug)]
struct Source{path:PathBuf,tensors:Vec<Tensor>,meta:Map<String,Value>,bytes:u64}
fn source(path:&Path)->io::Result<Source>{
    let path=path.canonicalize()?;
    let mut f=File::open(&path)?;let bytes=f.metadata()?.len();
    if bytes<10{return Err(bad("safetensors file too short"));}
    let mut n=[0;8];f.read_exact(&mut n)?;
    let header_len=u64::from_le_bytes(n);
    if header_len==0||header_len>HEADER_MAX||header_len>bytes-8{return Err(bad("invalid safetensors header"));}
    let mut buf=vec![0;header_len as usize];f.read_exact(&mut buf)?;
    let root:Value=serde_json::from_slice(&buf).map_err(|e|bad(format!("invalid header: {e}")))?;
    let root=root.as_object().ok_or_else(||bad("header is not an object"))?;
    if root.len()>100000{return Err(bad("too many source tensors"));}
    let meta=root.get("__metadata__").and_then(Value::as_object).cloned().unwrap_or_default();
    if meta.contains_key("ojon.format"){return Err(invalid("source already packed; use an original full precision checkpoint"));}
    let mut tensors=Vec::new();
    for (name,obj) in root {
        if name=="__metadata__"{continue;}
        if name.contains('\0')||name.is_empty(){return Err(bad("invalid tensor name"));}
        let dtype=obj["dtype"].as_str().ok_or_else(||bad("missing dtype"))?;
        let w=width(dtype).ok_or_else(||bad(format!("unsupported source dtype: {dtype}")))?;
        let shape=obj["shape"].as_array().ok_or_else(||bad("missing tensor shape"))?
            .iter().map(|x|x.as_u64().ok_or_else(||bad("invalid tensor dimension")))
            .collect::<io::Result<Vec<_>>>()?;
        if shape.len()>16{return Err(bad("tensor rank exceeds maximum"));}
        let off=obj["data_offsets"].as_array().ok_or_else(||bad("missing offsets"))?;
        if off.len()!=2{return Err(bad("malformed offsets"));}
        let a=off[0].as_u64().ok_or_else(||bad("invalid offset"))?;
        let b=off[1].as_u64().ok_or_else(||bad("invalid offset"))?;
        if b<a||b>bytes-8-header_len||b-a!=checked_len(&shape,w)?{
            return Err(bad("out-of-bounds or incorrect tensor length"));
        }
        if name.ends_with(".qdata")||name.ends_with(".qweight")||name.ends_with(".codebook"){
            return Err(invalid("source appears already quantized; refusing requantization"));
        }
        tensors.push(Tensor{name:name.clone(),dtype:dtype.to_owned(),shape,
            start:a+8+header_len,end:b+8+header_len});
    }
    if tensors.is_empty(){return Err(bad("no tensors"));}
    tensors.sort_by_key(|t|t.start);
    let mut prev=header_len+8;
    for t in &tensors{
        if t.start<prev{return Err(bad("overlapping tensor offsets"));}
        prev=t.end;
    }
    Ok(Source{path,tensors,meta,bytes})
}
fn protected(name:&str)->bool{
    let n=name.to_ascii_lowercase();
    ["embed","norm","bias","time_","time.","ada","modulation",
     "caption","text_","text.","patch_","proj_in","proj_out",
     "final_layer","rope","freqs","position","rotary","condition","lm_head"]
    .iter().any(|v|n.contains(v))
}
fn eligible(t:&Tensor)->bool{
    t.shape.len()==2&&t.shape[0]>0&&t.shape[1]>=64&&
    matches!(t.dtype.as_str(),"F16"|"BF16"|"F32")&&
    (t.name.ends_with(".weight")||t.name.ends_with("_weight"))&&!protected(&t.name)
}
#[derive(Clone,Debug)]
struct Entry {
    source:Tensor,quant:bool,offset:u64,length:u64,
    scale_offset:u64,bias_offset:u64,rows:usize,cols:usize,groups:usize,row_bytes:usize
}
#[derive(Debug)]
pub struct PackPlan {
    src:Source,output:PathBuf,bits:u8,group:usize,entries:Vec<Entry>,
    header:Vec<u8>,payload:u64,packed_count:usize
}
pub fn plan_pack(input:&Path,output:&Path,bits:u8,group:usize)->io::Result<PackPlan>{
    if !matches!(bits,2|3|4){return Err(invalid("bits must be 2, 3 or 4"));}
    if !(32..=256).contains(&group)||!group.is_power_of_two(){
        return Err(invalid("group must be 32, 64, 128 or 256"));
    }
    if output.extension().and_then(|x|x.to_str())!=Some("ojonq"){
        return Err(invalid("packed output needs .ojonq extension (not safetensors)"));
    }
    let src=source(input)?;
    let parent=output.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let output=parent.canonicalize()?.join(output.file_name().ok_or_else(||invalid("output filename missing"))?);
    if output==src.path||output.symlink_metadata().is_ok(){
        return Err(invalid("packed output already exists or equals source"));
    }
    let mut offset=0u64;let mut entries=Vec::new();let mut manifest=Vec::new();let mut packed=0;
    for t in &src.tensors{
        let quant=eligible(t);
        let (rows,cols,groups,row_bytes)=if quant{
            let r=usize::try_from(t.shape[0]).map_err(|_|bad("rows too large"))?;
            let c=usize::try_from(t.shape[1]).map_err(|_|bad("columns too large"))?;
            if c.checked_mul(width(&t.dtype).unwrap()).ok_or_else(||bad("row overflow"))?>ROW_MAX{
                return Err(bad("source row exceeds 16 MiB bounded-memory limit"));
            }
            (r,c,c.div_ceil(group),c.checked_mul(bits as usize).and_then(|n|n.checked_add(7))
                .ok_or_else(||bad("packed row overflow"))?/8)
        }else{(0,0,0,0)};
        let length=if quant{(rows as u64).checked_mul(row_bytes as u64).ok_or_else(||bad("packed length overflow"))?}
            else{t.end-t.start};
        let scale_offset=offset.checked_add(length).ok_or_else(||bad("output size overflow"))?;
        let overhead=(rows as u64).checked_mul(groups as u64).and_then(|x|x.checked_mul(4))
            .ok_or_else(||bad("group scale overflow"))?;
        let bias_offset=scale_offset.checked_add(overhead).ok_or_else(||bad("output size overflow"))?;
        let new_end=if quant{bias_offset.checked_add(overhead).ok_or_else(||bad("output size overflow"))?}
            else{scale_offset};
        manifest.push(json!({"name":t.name,"dtype":t.dtype,"shape":t.shape,"quantized":quant,
            "offset":offset,"length":length,"scale_offset":scale_offset,"bias_offset":bias_offset,
            "groups":groups,"row_bytes":row_bytes}));
        entries.push(Entry{source:t.clone(),quant,offset,length,scale_offset,bias_offset,
            rows,cols,groups,row_bytes});
        if quant{packed+=1;}
        offset=new_end;
    }
    if packed==0{return Err(invalid("no eligible 2D original full-precision weights to pack"));}
    let header=serde_json::to_vec(&json!({
        "format":"ojon-packed-v1","bits":bits,"group":group,"tensors":manifest,
        "source_metadata":src.meta,"loader":"ojon-unpack-required",
        "quality":"not_evaluated"
    })).map_err(|e|bad(format!("header serialization: {e}")))?;
    if header.len() as u64>HEADER_MAX{return Err(bad("packed header exceeds 32 MiB"));}
    Ok(PackPlan{src,output,bits,group,entries,header,payload:offset,packed_count:packed})
}
impl PackPlan{
    pub fn json(&self)->Value{
        json!({"schema":"ojon.pack_plan.v1","source":self.src.path.to_string_lossy(),
            "output":self.output.to_string_lossy(),"bits":self.bits,"group":self.group,
            "original_bytes":self.src.bytes,"estimated_output_bytes":16+self.header.len() as u64+self.payload,
            "packed_matrices":self.packed_count,"format":"ojon-packed-v1",
            "status":"PLAN_ONLY","gpu_loader":"NOT_IMPLEMENTED","task_quality":"NOT_EVALUATED"})
    }
}
fn f16_f32(h:u16)->f32{
    let sign=((h&0x8000) as u32)<<16;let exp=(h>>10)&31;let frac=(h&0x3ff) as u32;
    f32::from_bits(if exp==0{
        if frac==0{sign}else{
            let mut m=frac;let mut e=-14i32;
            while m&0x400==0{m<<=1;e-=1;}
            sign|(((e+127) as u32)<<23)|((m&0x3ff)<<13)
        }
    }else if exp==31{sign|0x7f800000|(frac<<13)}
    else{sign|((((exp as i32)-15+127) as u32)<<23)|(frac<<13)})
}
fn f32_f16(v:f32)->u16{
    let u=v.to_bits();let sign=((u>>16)&0x8000) as u16;
    let e=((u>>23)&255) as i32-127+15;let frac=u&0x7fffff;
    if e>=31{return sign|0x7c00;}
    if e<=0{
        if e< -10{return sign;}
        let m=frac|0x800000;let shift=(14-e) as u32;
        let low=m&((1u32<<shift)-1);let half=m>>shift;let midpoint=1u32<<(shift-1);
        return sign|((half+u32::from(low>midpoint||(low==midpoint&&half&1!=0))) as u16);
    }
    let mant=frac+0xfff+((frac>>13)&1);
    if mant>=0x800000{
        return if e==30{sign|0x7c00}else{sign|(((e+1) as u16)<<10)};
    }
    sign|((e as u16)<<10)|((mant>>13) as u16)
}
fn decode_float(buf:&[u8],dtype:&str)->io::Result<f32>{
    let v=match dtype{
        "F32"=>f32::from_le_bytes(buf.try_into().map_err(|_|bad("invalid f32"))?),
        "F16"=>f16_f32(u16::from_le_bytes(buf.try_into().map_err(|_|bad("invalid f16"))?)),
        "BF16"=>f32::from_bits((u16::from_le_bytes(buf.try_into().map_err(|_|bad("invalid bf16"))?) as u32)<<16),
        _=>return Err(bad("unsupported quantized dtype"))
    };
    if !v.is_finite(){return Err(bad("nonfinite input weight cannot be quantized safely"));}
    Ok(v)
}
fn encode_float(v:f32,dtype:&str)->io::Result<Vec<u8>>{
    Ok(match dtype{
        "F32"=>v.to_le_bytes().to_vec(),
        "F16"=>f32_f16(v).to_le_bytes().to_vec(),
        "BF16"=>{let x=v.to_bits();(((x+0x7fff+((x>>16)&1))>>16) as u16).to_le_bytes().to_vec()},
        _=>return Err(bad("unsupported unpacked dtype"))
    })
}
fn pack_codes(codes:&[u8],bits:u8,out:&mut [u8]){
    out.fill(0);
    for (i,code) in codes.iter().enumerate(){
        let p=i*bits as usize;
        out[p/8]|=*code<<(p%8);
        if p%8+bits as usize>8{out[p/8+1]|=*code>>(8-p%8);}
    }
}
fn unpack_code(bytes:&[u8],bits:u8,index:usize)->u8{
    let pos=index*bits as usize;
    let mut v=(bytes[pos/8] as u16)>>(pos%8);
    if pos%8+bits as usize>8{v|=(bytes[pos/8+1] as u16)<<(8-pos%8);}
    (v&((1u16<<bits)-1)) as u8
}
fn quant_group(values:&[f32],codes:&mut [u8],bits:u8)->(f32,f32,f64){
    let lo=values.iter().copied().fold(f32::INFINITY,f32::min);
    let hi=values.iter().copied().fold(f32::NEG_INFINITY,f32::max);
    let levels=((1u32<<bits)-1) as f32;
    let scale=if hi>lo{(hi-lo)/levels}else{1.0};
    let mut mse=0.0;
    for (i,&value) in values.iter().enumerate(){
        let code=((value-lo)/scale).round().clamp(0.0,levels) as u8;
        codes[i]=code;
        let error=(value-(lo+(code as f32)*scale)) as f64;
        mse+=error*error;
    }
    (scale,lo,mse)
}
struct Stage(PathBuf);
impl Drop for Stage{fn drop(&mut self){let _=fs::remove_dir_all(&self.0);}}
fn stage(parent:&Path)->io::Result<Stage>{
    for n in 0..8{
        let now=SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_|bad("clock"))?.as_nanos();
        let p=parent.join(format!(".ojon-native-{}-{now}-{n}",std::process::id()));
        match fs::create_dir(&p){
            Ok(())=>return Ok(Stage(p)),
            Err(e) if e.kind()==io::ErrorKind::AlreadyExists=>continue,
            Err(e)=>return Err(e)
        }
    }
    Err(bad("cannot create staging directory"))
}
fn read_at(f:&mut File,offset:u64,buf:&mut [u8])->io::Result<()>{
    f.seek(SeekFrom::Start(offset))?;f.read_exact(buf)
}
fn write_at(f:&mut File,offset:u64,buf:&[u8])->io::Result<()>{
    f.seek(SeekFrom::Start(offset))?;f.write_all(buf)
}
fn copy_at(src:&mut File,dst:&mut File,from:u64,to:u64,len:u64)->io::Result<()>{
    src.seek(SeekFrom::Start(from))?;dst.seek(SeekFrom::Start(to))?;
    let count=io::copy(&mut src.take(len),dst)?;
    if count!=len{return Err(bad("short input tensor"));}
    Ok(())
}
fn write_prefix(f:&mut File,header:&[u8],payload:u64)->io::Result<u64>{
    f.write_all(MAGIC)?;
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header)?;
    let offset=16+header.len() as u64;
    f.set_len(offset.checked_add(payload).ok_or_else(||bad("output too large"))?)?;
    Ok(offset)
}
#[derive(Debug)]
pub struct PackResult{pub output:PathBuf,pub output_bytes:u64,pub original_bytes:u64,
    pub packed_tensors:usize,pub elements:u64,pub mse:f64}
pub fn run_pack<F:FnMut(&str)>(plan:&PackPlan,mut log:F)->io::Result<PackResult>{
    if plan.output.symlink_metadata().is_ok(){return Err(io::Error::new(io::ErrorKind::AlreadyExists,"output exists"));}
    let dir=stage(plan.output.parent().ok_or_else(||bad("missing output parent"))?)?;
    let file=dir.0.join("model.ojonq");
    let mut dest=OpenOptions::new().create_new(true).write(true).read(true).open(&file)?;
    let base=write_prefix(&mut dest,&plan.header,plan.payload)?;
    let mut original=File::open(&plan.src.path)?;
    let mut squared=0.0;let mut count=0u64;
    for entry in &plan.entries {
        if !entry.quant{
            copy_at(&mut original,&mut dest,entry.source.start,base+entry.offset,entry.length)?;
            continue;
        }
        let elem=width(&entry.source.dtype).unwrap();
        let row_len=entry.cols*elem;
        let mut raw=vec![0;row_len];
        let mut values=vec![0f32;entry.cols];
        let mut codes=vec![0u8;entry.cols];
        let mut packed=vec![0u8;entry.row_bytes];
        let mut scales=vec![0;entry.groups*4];
        let mut bias=vec![0;entry.groups*4];
        for row in 0..entry.rows {
            read_at(&mut original,entry.source.start+(row*row_len) as u64,&mut raw)?;
            for (i,chunk) in raw.chunks_exact(elem).enumerate(){
                values[i]=decode_float(chunk,&entry.source.dtype)?;
            }
            for group in 0..entry.groups {
                let a=group*plan.group;
                let b=std::cmp::min(entry.cols,a+plan.group);
                let (s,o,err)=quant_group(&values[a..b],&mut codes[a..b],plan.bits);
                scales[4*group..4*group+4].copy_from_slice(&s.to_le_bytes());
                bias[4*group..4*group+4].copy_from_slice(&o.to_le_bytes());
                squared+=err;
            }
            pack_codes(&codes,plan.bits,&mut packed);
            write_at(&mut dest,base+entry.offset+(row*entry.row_bytes) as u64,&packed)?;
            write_at(&mut dest,base+entry.scale_offset+(row*entry.groups*4) as u64,&scales)?;
            write_at(&mut dest,base+entry.bias_offset+(row*entry.groups*4) as u64,&bias)?;
            count+=entry.cols as u64;
        }
        log(&format!("[ojon] packed {} at {} bits",entry.source.name,plan.bits));
    }
    dest.sync_all()?;drop(dest);
    // Verify magic/header length and the final on-disk size before publication.
    let mut verify=File::open(&file)?;
    let mut prefix=[0;16];verify.read_exact(&mut prefix)?;
    if &prefix[..8]!=MAGIC||u64::from_le_bytes(prefix[8..16].try_into().unwrap())!=plan.header.len() as u64{
        return Err(bad("packed container failed verification"));
    }
    let actual=verify.metadata()?.len();
    if actual!=base+plan.payload{return Err(bad("packed container truncated"));}
    fs::hard_link(&file,&plan.output)?;
    Ok(PackResult{output:plan.output.clone(),output_bytes:actual,original_bytes:plan.src.bytes,
        packed_tensors:plan.packed_count,elements:count,
        mse:if count>0{squared/count as f64}else{0.0}})
}
/// Data decoding is checked and produces conventional safetensors. The normal
/// GPU pipeline loads restored full-precision tensors, NOT packed weights.
pub fn unpack(input:&Path,output:&Path)->io::Result<PathBuf>{
    let input=input.canonicalize()?;
    let mut src=File::open(&input)?;
    let size=src.metadata()?.len();
    if size<18{return Err(bad("packed file too short"));}
    let mut prefix=[0u8;16];src.read_exact(&mut prefix)?;
    if &prefix[..8]!=MAGIC{return Err(bad("not an Ojon packed container"));}
    let hlen=u64::from_le_bytes(prefix[8..16].try_into().unwrap());
    if hlen==0||hlen>HEADER_MAX||hlen>size-16{return Err(bad("bad packed header length"));}
    let mut h=vec![0;hlen as usize];src.read_exact(&mut h)?;
    let root:Value=serde_json::from_slice(&h).map_err(|e|bad(format!("bad packed JSON: {e}")))?;
    if root["format"].as_str()!=Some("ojon-packed-v1"){return Err(bad("unknown container version"));}
    let bits=root["bits"].as_u64().ok_or_else(||bad("missing bits"))? as u8;
    let group=root["group"].as_u64().ok_or_else(||bad("missing group"))? as usize;
    if !matches!(bits,2|3|4)||!(32..=256).contains(&group)||!group.is_power_of_two(){
        return Err(bad("unsupported packed bitwidth or grouping"));
    }
    let items=root["tensors"].as_array().ok_or_else(||bad("missing packed tensors"))?;
    if items.len()>100000{return Err(bad("too many packed tensors"));}
    let base=16+hlen;
    let parent=output.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let output=parent.canonicalize()?.join(output.file_name().ok_or_else(||invalid("missing output filename"))?);
    if output==input||output.symlink_metadata().is_ok()||output.extension().and_then(|s|s.to_str())!=Some("safetensors"){
        return Err(invalid("unpack needs a new .safetensors output"));
    }
    let mut desc=Map::new();let mut cursor=0u64;
    for item in items{
        let name=item["name"].as_str().ok_or_else(||bad("bad tensor name"))?;
        let dtype=item["dtype"].as_str().ok_or_else(||bad("bad original dtype"))?;
        let w=width(dtype).ok_or_else(||bad("unsupported original dtype"))?;
        let shape=item["shape"].as_array().ok_or_else(||bad("missing original shape"))?
            .iter().map(|x|x.as_u64().ok_or_else(||bad("bad tensor dimension")))
            .collect::<io::Result<Vec<_>>>()?;
        let len=checked_len(&shape,w)?;
        let start=cursor;cursor=cursor.checked_add(len).ok_or_else(||bad("restored model too large"))?;
        if desc.insert(name.to_owned(),json!({"dtype":dtype,"shape":shape,"data_offsets":[start,cursor]})).is_some(){
            return Err(bad("duplicate tensor name"));
        }
    }
    let metadata=root.get("source_metadata").and_then(Value::as_object).cloned().unwrap_or_default();
    if !metadata.is_empty(){desc.insert("__metadata__".into(),Value::Object(metadata));}
    let header=serde_json::to_vec(&desc).map_err(|e|bad(format!("safetensors header error: {e}")))?;
    if header.len() as u64>HEADER_MAX{return Err(bad("restored header too big"));}
    let stage=stage(output.parent().ok_or_else(||bad("missing parent"))?)?;
    let file=stage.0.join("restored.safetensors");
    let mut dst=OpenOptions::new().create_new(true).write(true).read(true).open(&file)?;
    dst.write_all(&(header.len() as u64).to_le_bytes())?;
    dst.write_all(&header)?;
    let target=8+header.len() as u64;
    dst.set_len(target+cursor)?;
    let mut to=0u64;
    for item in items{
        let shape=item["shape"].as_array().unwrap();
        let dtype=item["dtype"].as_str().unwrap();let w=width(dtype).unwrap();
        let len=shape.iter().try_fold(w as u64,|v,x|v.checked_mul(x.as_u64().unwrap()).ok_or_else(||bad("size overflow")))?;
        let from=item["offset"].as_u64().ok_or_else(||bad("missing packed offset"))?;
        let packed=item["quantized"].as_bool().ok_or_else(||bad("missing quantized flag"))?;
        if !packed{
            if from.checked_add(len).ok_or_else(||bad("overflow"))?>size-base{return Err(bad("packed bytes outside file"));}
            copy_at(&mut src,&mut dst,base+from,target+to,len)?;
        }else{
            if shape.len()!=2{return Err(bad("quantized tensor rank must be 2"));}
            let rows=usize::try_from(shape[0].as_u64().unwrap()).map_err(|_|bad("rows overflow"))?;
            let cols=usize::try_from(shape[1].as_u64().unwrap()).map_err(|_|bad("cols overflow"))?;
            let groups=cols.div_ceil(group);
            let rb=cols.checked_mul(bits as usize).and_then(|v|v.checked_add(7)).ok_or_else(||bad("row overflow"))?/8;
            if cols.checked_mul(w).ok_or_else(||bad("row overflow"))?>ROW_MAX{return Err(bad("row too wide"));}
            let so=item["scale_offset"].as_u64().ok_or_else(||bad("missing scales"))?;
            let bo=item["bias_offset"].as_u64().ok_or_else(||bad("missing biases"))?;
            let clen=(rows as u64).checked_mul(rb as u64).ok_or_else(||bad("size overflow"))?;
            let flen=(rows as u64).checked_mul(groups as u64).and_then(|v|v.checked_mul(4)).ok_or_else(||bad("size overflow"))?;
            if from+clen>size-base||so+flen>size-base||bo+flen>size-base{
                return Err(bad("packed tensor outside container"));
            }
            let mut q=vec![0;rb];let mut sc=vec![0;groups*4];let mut bi=vec![0;groups*4];
            let mut restored=vec![0;cols*w];
            for row in 0..rows {
                read_at(&mut src,base+from+(row*rb) as u64,&mut q)?;
                read_at(&mut src,base+so+(row*groups*4) as u64,&mut sc)?;
                read_at(&mut src,base+bo+(row*groups*4) as u64,&mut bi)?;
                for col in 0..cols{
                    let g=col/group;
                    let scale=f32::from_le_bytes(sc[g*4..g*4+4].try_into().unwrap());
                    let bias=f32::from_le_bytes(bi[g*4..g*4+4].try_into().unwrap());
                    if !scale.is_finite()||scale<=0.0||!bias.is_finite(){return Err(bad("bad packed scale/bias"));}
                    let value=bias+(unpack_code(&q,bits,col) as f32)*scale;
                    let enc=encode_float(value,dtype)?;
                    restored[col*w..(col+1)*w].copy_from_slice(&enc);
                }
                write_at(&mut dst,target+to+(row*restored.len()) as u64,&restored)?;
            }
        }
        to+=len;
    }
    dst.sync_all()?;drop(dst);
    source(&file)?; // full safetensors shape/offset validation before publish
    fs::hard_link(&file,&output)?;
    Ok(output)
}
#[cfg(test)]
mod tests{
    use super::*;
    struct Temp(PathBuf);
    impl Drop for Temp{fn drop(&mut self){let _=fs::remove_dir_all(&self.0);}}
    fn temp()->Temp{
        let n=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let p=std::env::temp_dir().join(format!("ojon-weights-{}-{n}",std::process::id()));
        fs::create_dir(&p).unwrap();Temp(p)
    }
    fn sample_file(p:&Path){
        let meta=json!({
            "blocks.0.attn.weight":{"dtype":"F32","shape":[8,128],"data_offsets":[0,4096]},
            "norm.weight":{"dtype":"F32","shape":[128],"data_offsets":[4096,4608]}
        }).to_string();
        let mut f=File::create(p).unwrap();
        f.write_all(&(meta.len() as u64).to_le_bytes()).unwrap();
        f.write_all(meta.as_bytes()).unwrap();
        for i in 0..1024{
            let v=(i as f32/13.0).sin()+((i%7) as f32)*0.01;
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for i in 0..128{f.write_all(&((i as f32)*0.01).to_le_bytes()).unwrap();}
    }
    #[test] fn pack_unpack_all_low_bit_formats(){
        let dir=temp();let source_path=dir.0.join("orig.safetensors");sample_file(&source_path);
        for bits in [2,3,4]{
            let packed=dir.0.join(format!("{bits}.ojonq"));
            let back=dir.0.join(format!("{bits}.safetensors"));
            let plan=plan_pack(&source_path,&packed,bits,64).unwrap();
            assert_eq!(plan.packed_count,1);assert!(!packed.exists());
            let result=run_pack(&plan,|_|{}).unwrap();
            assert_eq!(result.elements,1024);assert!(result.mse.is_finite());
            let restored=unpack(&packed,&back).unwrap();
            let original=source(&source_path).unwrap();
            let copy=source(&restored).unwrap();
            assert_eq!(copy.tensors.len(),original.tensors.len());
            let a=original.tensors.iter().find(|x|x.name=="norm.weight").unwrap();
            let b=copy.tensors.iter().find(|x|x.name=="norm.weight").unwrap();
            let mut f1=File::open(&source_path).unwrap();let mut f2=File::open(&restored).unwrap();
            let mut x=vec![0;512];let mut y=vec![0;512];
            read_at(&mut f1,a.start,&mut x).unwrap();read_at(&mut f2,b.start,&mut y).unwrap();
            assert_eq!(x,y);
            assert!(plan_pack(&source_path,&packed,bits,64).is_err());
        }
    }
    #[test] fn float16_and_codes_roundtrip(){
        for v in [-3.0,-1.0,0.0,0.5,1.0,70.0]{
            assert!((f16_f32(f32_f16(v))-v).abs()<0.1);
        }
        for bits in [2,3,4]{
            let values=[-1.0,0.0,1.0,0.5];
            let mut codes=[0;4];let (_,_,err)=quant_group(&values,&mut codes,bits);
            assert!(err.is_finite());
            let mut buf=vec![0;(values.len()*bits as usize).div_ceil(8)];
            pack_codes(&codes,bits,&mut buf);
            for i in 0..4{assert_eq!(codes[i],unpack_code(&buf,bits,i));}
        }
    }
}
