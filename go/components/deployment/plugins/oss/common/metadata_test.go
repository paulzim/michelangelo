package common

import (
	"testing"

	"github.com/gogo/protobuf/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
)

func TestReadModelLoadedFlag_NilMetadata(t *testing.T) {
	got, err := ReadModelLoadedFlag(&apipb.Condition{})
	require.NoError(t, err)
	assert.False(t, got)
}

func TestReadModelLoadedFlag_MalformedMetadata(t *testing.T) {
	// Wrap a non-BoolValue type so UnmarshalAny on BoolValue fails.
	bogus, err := types.MarshalAny(&types.StringValue{Value: "not a bool"})
	require.NoError(t, err)

	_, err = ReadModelLoadedFlag(&apipb.Condition{Metadata: bogus})
	assert.Error(t, err)
}

func TestModelLoadedFlag_RoundTrip(t *testing.T) {
	condition := &apipb.Condition{}
	require.NoError(t, WriteModelLoadedFlag(condition))
	require.NotNil(t, condition.Metadata)

	got, err := ReadModelLoadedFlag(condition)
	require.NoError(t, err)
	assert.True(t, got)
}
