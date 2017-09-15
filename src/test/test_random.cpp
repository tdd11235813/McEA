#include "catch.hpp"
#include <random>

#include "../util/random.cu"

TEST_CASE( "host random numbers in right range [0, 1)", "[random]" ) {

  for (int i = 0; i < 1000; ++i) {
    float rn = randomFloat();
    REQUIRE( rn < 1.0 );
    REQUIRE( rn >= 0.0 );
  }
}

TEST_CASE( "PRNG initialized correctly", "[random]" ) {

  std::default_random_engine prng = rand_init( 0l );

  REQUIRE_NOTHROW( prng() );
}

TEST_CASE( "transformation of float to uniform integer in a range", "[random]" ) {

  REQUIRE( trans_uniform_int( 0.0, 5 ) == 0 );
  REQUIRE( trans_uniform_int( 0.21, 5 ) == 1 );
  REQUIRE( trans_uniform_int( 0.42, 5 ) == 2 );
  REQUIRE( trans_uniform_int( 0.67, 5 ) == 3 );
  REQUIRE( trans_uniform_int( 0.81, 5 ) == 4 );
  REQUIRE( trans_uniform_int( 1.0, 5 ) == 4 );

  REQUIRE( trans_uniform_int( 0.5, 1 ) == 0 );
  REQUIRE( trans_uniform_int( 0.5, 2 ) == 0 );
  REQUIRE( trans_uniform_int( 0.50001, 2 ) == 1 );
  REQUIRE( trans_uniform_int( 0.5, 85 ) == 42 );
}
