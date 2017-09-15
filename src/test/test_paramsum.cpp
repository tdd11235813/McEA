#include "catch.hpp"

#define DTLZ_NUM 0

#include "../util/dtlz.cu"

TEST_CASE("paramsum example calculates right", "[dtlz]") {
  int p_size = 10;
  int o_size = 3;
  float obj[3];

  float params0[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
  dtlz( params0, obj, p_size, o_size, 1 );
  REQUIRE( obj[0] == Approx( 0.0 ) );
  REQUIRE( obj[1] == Approx( 0.0 ) );
  REQUIRE( obj[2] == Approx( 0.0 ) );

  float params1[] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; 
  dtlz( params1, obj, p_size, o_size, 1 );
  REQUIRE( obj[0] == Approx( 0.0 ) );
  REQUIRE( obj[1] == Approx( 10.0 ) );
  REQUIRE( obj[2] == Approx( 20.0 ) );

  float params05[] = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}; 
  dtlz( params05, obj, p_size, o_size, 1 );
  REQUIRE( obj[0] == Approx( 0.0 ) );
  REQUIRE( obj[1] == Approx( 5.0 ) );
  REQUIRE( obj[2] == Approx( 10.0 ) );
}
