<script lang="ts">
  import SubTabs from '../SubTabs.svelte'
  import { subRoute } from '../router'
  import Simulator from './Simulator.svelte'
  import Patient from './Patient.svelte'
</script>

<SubTabs top="sim" />
{#if $subRoute === 'explore'}
  <Patient />
{:else}
  <Simulator />
{/if}
