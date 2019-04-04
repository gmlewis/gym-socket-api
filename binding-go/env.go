package gym

import (
	"bufio"
	"encoding/json"
	"errors"
	"net"
	"path/filepath"
	"sync"

	"github.com/unixpickle/essentials"
)

// Env is a handle on a Gym environment.
//
// The methods on an Env are thread-safe.
type Env interface {
	// Reset resets the environment.
	Reset() (obs Obs, err error)

	// Step takes an action.
	Step(action interface{}) (obs Obs, reward float64,
		done bool, info interface{}, err error)

	// ActionSpace gets the action space.
	ActionSpace() (*Space, error)

	// ObservationSpace gets the observation space.
	ObservationSpace() (*Space, error)

	// SampleAction samples from the action space.
	//
	// The action is written to dst in the same way
	// that Obs.Unmarshal() does it.
	SampleAction(dst interface{}) error

	// Monitor sets the environment up to save results
	// to the given directory.
	//
	// If the directory is a relative path, it should be
	// relative to the current working directory.
	Monitor(dir string, force, resume, video bool) error

	// Render graphically renders the environment.
	Render() error

	// Close stops and cleans up the environment.
	Close() error

	// UniverseConfigure configures a Universe environment.
	//
	// The options argument may be nil.
	UniverseConfigure(options map[string]interface{}) error

	// UniverseWrap wraps a Universe environment.
	//
	// The options argument may be nil.
	UniverseWrap(wrapper string, options map[string]interface{}) error

	// RetroConfigure configures a Retro environment.
	//
	// The options argument may be nil.
	RetroConfigure(options map[string]interface{}) error

	// RetroWrap wraps a Retro environment.
	//
	// The options argument may be nil.
	RetroWrap(wrapper string, options map[string]interface{}) error
}

type connEnv struct {
	Buf  *bufio.ReadWriter
	Conn net.Conn

	CmdLock sync.Mutex
}

// Make creates an Env by connecting to an API server and
// requesting the given environment.
func Make(host, envName string) (env Env, err error) {
	defer essentials.AddCtxTo("make environment", &err)
	conn, err := net.Dial("tcp", host)
	if err != nil {
		return nil, err
	}

	rw := bufio.NewReadWriter(bufio.NewReader(conn), bufio.NewWriter(conn))
	if err := handshake(rw, envName); err != nil {
		conn.Close()
		return nil, err
	}

	return &connEnv{Buf: rw, Conn: conn}, nil
}

func (c *connEnv) Reset() (obs Obs, err error) {
	defer essentials.AddCtxTo("reset environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetReset); err != nil {
		return nil, err
	}
	if err := c.Buf.Flush(); err != nil {
		return nil, err
	}
	return readObservation(c.Buf)
}

func (c *connEnv) Step(action interface{}) (obs Obs, reward float64,
	done bool, info interface{}, err error) {
	defer essentials.AddCtxTo("step environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	err = writePacketType(c.Buf, packetStep)
	if err != nil {
		return
	}
	err = writeAction(c.Buf, action)
	if err != nil {
		return
	}
	err = c.Buf.Flush()
	if err != nil {
		return
	}
	obs, err = readObservation(c.Buf)
	if err != nil {
		return
	}
	reward, err = readReward(c.Buf)
	if err != nil {
		return
	}
	done, err = readBool(c.Buf)
	if err != nil {
		return
	}
	infoData, err := readByteField(c.Buf)
	if err != nil {
		return
	}
	err = json.Unmarshal(infoData, &info)
	return
}

func (c *connEnv) ActionSpace() (*Space, error) {
	return c.getSpace(actionSpace)
}

func (c *connEnv) ObservationSpace() (*Space, error) {
	return c.getSpace(observationSpace)
}

func (c *connEnv) SampleAction(dst interface{}) (err error) {
	essentials.AddCtxTo("sample action", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetSampleAction); err != nil {
		return err
	}
	if err := c.Buf.Flush(); err != nil {
		return err
	}
	return readAction(c.Buf, dst)
}

func (c *connEnv) Monitor(dir string, force, resume, video bool) (err error) {
	essentials.AddCtxTo("monitor environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetMonitor); err != nil {
		return err
	}
	for _, b := range []bool{resume, force, video} {
		if err := writeBool(c.Buf, b); err != nil {
			return err
		}
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if err := writeByteField(c.Buf, []byte(absDir)); err != nil {
		return err
	}
	if err := c.Buf.Flush(); err != nil {
		return err
	}
	if errData, err := readByteField(c.Buf); err != nil {
		return err
	} else if len(errData) > 0 {
		return errors.New(string(errData))
	}
	return nil
}

func (c *connEnv) Render() (err error) {
	essentials.AddCtxTo("render environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetRender); err != nil {
		return err
	}
	return c.Buf.Flush()
}

func (c *connEnv) Close() error {
	return c.Conn.Close()
}

func (c *connEnv) UniverseConfigure(options map[string]interface{}) (err error) {
	if options == nil {
		options = map[string]interface{}{}
	}
	essentials.AddCtxTo("configure Universe environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetUniverseConfigure); err != nil {
		return err
	}
	jsonData, err := json.Marshal(options)
	if err != nil {
		return err
	}
	if err := writeByteField(c.Buf, jsonData); err != nil {
		return err
	}
	if err := c.Buf.Flush(); err != nil {
		return err
	}
	return readErrorField(c.Buf)
}

func (c *connEnv) UniverseWrap(wrapper string,
	options map[string]interface{}) (err error) {
	if options == nil {
		options = map[string]interface{}{}
	}
	essentials.AddCtxTo("wrap Universe environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetUniverseWrap); err != nil {
		return err
	}
	if err := writeByteField(c.Buf, []byte(wrapper)); err != nil {
		return err
	}
	jsonData, err := json.Marshal(options)
	if err != nil {
		return err
	}
	if err := writeByteField(c.Buf, jsonData); err != nil {
		return err
	}
	if err := c.Buf.Flush(); err != nil {
		return err
	}
	return readErrorField(c.Buf)
}

func (c *connEnv) RetroConfigure(options map[string]interface{}) (err error) {
	if options == nil {
		options = map[string]interface{}{}
	}
	essentials.AddCtxTo("configure Retro environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetRetroConfigure); err != nil {
		return err
	}
	jsonData, err := json.Marshal(options)
	if err != nil {
		return err
	}
	if err := writeByteField(c.Buf, jsonData); err != nil {
		return err
	}
	if err := c.Buf.Flush(); err != nil {
		return err
	}
	return readErrorField(c.Buf)
}

func (c *connEnv) RetroWrap(wrapper string,
	options map[string]interface{}) (err error) {
	if options == nil {
		options = map[string]interface{}{}
	}
	essentials.AddCtxTo("wrap Retro environment", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetRetroWrap); err != nil {
		return err
	}
	if err := writeByteField(c.Buf, []byte(wrapper)); err != nil {
		return err
	}
	jsonData, err := json.Marshal(options)
	if err != nil {
		return err
	}
	if err := writeByteField(c.Buf, jsonData); err != nil {
		return err
	}
	if err := c.Buf.Flush(); err != nil {
		return err
	}
	return readErrorField(c.Buf)
}

func (c *connEnv) getSpace(spaceID int) (space *Space, err error) {
	essentials.AddCtxTo("get space info", &err)
	c.CmdLock.Lock()
	defer c.CmdLock.Unlock()
	if err := writePacketType(c.Buf, packetGetSpace); err != nil {
		return nil, err
	}
	if err := writeSpaceType(c.Buf, spaceID); err != nil {
		return nil, err
	}
	if err := c.Buf.Flush(); err != nil {
		return nil, err
	}
	data, err := readByteField(c.Buf)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &space); err != nil {
		return nil, err
	}
	return
}
