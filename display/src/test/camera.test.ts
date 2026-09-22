import { describe, expect, it, vi } from 'vitest';
import { CameraController } from '../camera';
import { CAPTURE_FPS, captureDimensions, FrameSender, MAX_BUFFERED } from '../ingest';
import { AnchorSmoother, coverPoint, overlayTarget } from '../placement';
import { readConfig } from '../config';

function stream(facing = 'user') {
  const track = { stop: vi.fn(), onended: null, onmute: null, onunmute: null, getSettings: () => ({ facingMode: facing }) };
  return { track, value: { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream };
}
function media(fn: ReturnType<typeof vi.fn>) { return { getUserMedia: fn } as unknown as MediaDevices; }
describe('local camera lifecycle', () => {
  it('starts only explicitly, requests no audio, prefers front and stops tracks', async () => {
    const fake = stream(), get = vi.fn().mockResolvedValue(fake.value);
    const c = new CameraController(media(get));
    expect(get).not.toHaveBeenCalled(); await c.start();
    expect(get).toHaveBeenCalledWith(expect.objectContaining({ audio: false, video: expect.objectContaining({ facingMode: { ideal: 'user' } }) }));
    expect(c.getState()).toMatchObject({status:'active',facing:'user'});
    c.stop(); expect(fake.track.stop).toHaveBeenCalledOnce(); expect(c.getState().stream).toBeNull();
  });
  it.each([['NotAllowedError','denied'],['NotFoundError','missing'],['NotReadableError','busy']])('distinguishes %s',async (name,status) => {
    const c = new CameraController(media(vi.fn().mockRejectedValue(new DOMException('private',name))));
    await c.start(); expect(c.getState().status).toBe(status);
  });
  it('releases the old camera before switching and handles interrupted tracks', async () => {
    const front = stream(), rear = stream('environment');
    const get = vi.fn().mockResolvedValueOnce(front.value).mockImplementationOnce(() => {
      expect(front.track.stop).toHaveBeenCalledOnce(); return Promise.resolve(rear.value);
    });
    const c = new CameraController(media(get)); await c.start(); await c.switchCamera();
    expect(c.getState().facing).toBe('environment');

    const ended = rear.track.onended as unknown as () => void; ended();
    expect(c.getState().status).toBe('interrupted'); expect(rear.track.stop).toHaveBeenCalledOnce();
  });
  it('disposes a permission result arriving after unmount', async () => {
    const fake = stream(); let resolve!: (value: MediaStream) => void;
    const c = new CameraController(media(vi.fn().mockReturnValue(new Promise<MediaStream>(r => {resolve=r;}))));
    const pending = c.start(); c.stop(); resolve(fake.value); await pending;
    expect(fake.track.stop).toHaveBeenCalledOnce(); expect(c.getState().status).toBe('idle');
  });
});
class Socket {
  readyState = 1; bufferedAmount = 0; send = vi.fn(); close = vi.fn();
  onopen: WebSocket['onopen'] = null; onmessage: WebSocket['onmessage'] = null;
  onerror: WebSocket['onerror'] = null; onclose: WebSocket['onclose'] = null;
  ready() { this.onopen?.call(this as unknown as WebSocket, {} as Event); this.onmessage?.call(this as unknown as WebSocket, {data:JSON.stringify({type:'ready',session_id:'test',max_frame_bytes:524288,max_fps:12})} as MessageEvent); }
}
describe('bounded authenticated frame sender', () => {
  it('does not encode before readiness, rate limits and drops on congestion', async () => {
    vi.useFakeTimers(); const socket = new Socket(), encode = vi.fn().mockResolvedValue(new Blob(['image'])); let now=0;
    const sender = new FrameSender(encode,()=>socket,()=>now);
    await sender.tick(); expect(encode).not.toHaveBeenCalled();
    sender.connect('ws://localhost/api/v1/camera/ingest','temporary-test-token');
    await sender.tick(); expect(encode).not.toHaveBeenCalled();
    socket.ready(); expect(JSON.parse(socket.send.mock.calls[0][0]).token).toBe('temporary-test-token');
    await sender.tick(); await sender.tick(); expect(encode).toHaveBeenCalledTimes(1);
    now+=1000/CAPTURE_FPS; socket.bufferedAmount=MAX_BUFFERED+1;
    await sender.tick(); expect(sender.getState().dropped).toBe(1);
    sender.disconnect(); now+=1000; await sender.tick(); expect(encode).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
  it('discards an encode finishing after disconnect', async () => {
    vi.useFakeTimers(); const socket = new Socket(); let resolve!: (value:Blob) => void;
    const sender = new FrameSender(()=>new Promise(r=>{resolve=r;}),()=>socket,()=>0);
    sender.connect('ws://localhost/api/v1/camera/ingest','temporary-test-token'); socket.ready();
    const pending=sender.tick(); sender.disconnect(); resolve(new Blob(['frame'])); await pending;
    expect(socket.send).toHaveBeenCalledTimes(1); expect(vi.getTimerCount()).toBe(0);
  });
  it('preserves aspect ratio for landscape and portrait inference', () => {
    expect(captureDimensions(1920,1080)).toEqual({width:640,height:360});
    expect(captureDimensions(1080,1920)).toEqual({width:203,height:360});
    expect(readConfig({},'?demo=1&camera=1')).toMatchObject({demo:true,cameraDemo:true});
  });
});
describe('cover-cropped person-relative placement', () => {
  it('maps image centre and corrects front mirroring after cropping', () => {
    expect(coverPoint({x:.5,y:.5},640,360,360,800,false)).toEqual({x:180,y:400});
    const front=coverPoint({x:.6,y:.4},640,360,360,800,true);
    const rear=coverPoint({x:.6,y:.4},640,360,360,800,false);
    expect(front.x+rear.x).toBeCloseTo(360); expect(front.y).toBeCloseTo(320);
    expect(coverPoint({x:.5,y:.5},360,640,844,390,false)).toEqual({x:422,y:195});
  });
  it('clamps every requested viewport and falls back for missing tracks', () => {
    for(const [width,height] of [[320,568],[360,800],[390,844],[412,915],[844,390],[1366,768]]) {
      const view={width,height,top:64,bottom:70,left:12,right:12};
      for(const box of [null,{x1:0,y1:0,x2:1,y2:1},{x1:.3,y1:.5,x2:.6,y2:.9}]) {
        const target=overlayTarget(box,640,360,view,true,150);
        expect(target.x).toBeGreaterThanOrEqual(99); expect(target.x).toBeLessThanOrEqual(width-99);
        expect(target.y).toBeGreaterThanOrEqual(120); expect(target.y).toBeLessThanOrEqual(height-142);
      }
    }
  });
  it('smooths with bounded velocity and clamps a suspended interval', () => {
    const smoother=new AnchorSmoother(); smoother.step({x:100,y:100},0);
    const next=smoother.step({x:900,y:900},1000);
    expect(Math.hypot(next.x-100,next.y-100)).toBeLessThanOrEqual(26.001);
    expect(next.x).toBeGreaterThan(100);
  });
  it('chooses the inward side of the displayed box when mirrored', () => {
    const view={width:1000,height:600,top:64,bottom:70,left:12,right:12};
    const box={x1:.1,y1:.05,x2:.3,y2:.8};
    const rear=overlayTarget(box,1000,600,view,false,150);
    const front=overlayTarget(box,1000,600,view,true,150);
    expect(rear.x).toBeGreaterThan(300);
    expect(front.x).toBeLessThan(700);
    expect(front.x+rear.x).toBeCloseTo(1000);
  });
});

